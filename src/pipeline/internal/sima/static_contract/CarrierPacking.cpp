#define SIMA_NEAT_INTERNAL 1
#include "pipeline/internal/sima/static_contract/CarrierPacking.h"

#include <algorithm>
#include <array>
#include <limits>
#include <unordered_set>

namespace simaai::neat::pipeline_internal::sima::static_contract {
namespace {
namespace client = ::simaai::scheduler::client;

bool fail(std::string* error, const char* message) {
  if (error)
    *error = message;
  return false;
}

bool is_power_of_two(std::uint64_t value) {
  return value && !(value & (value - 1U));
}

bool add(std::uint64_t lhs, std::uint64_t rhs, std::uint64_t& result) {
  if (lhs > std::numeric_limits<std::uint64_t>::max() - rhs)
    return false;
  result = lhs + rhs;
  return true;
}

std::optional<std::uint64_t> align_up(std::uint64_t value, std::uint64_t alignment) {
  if (!is_power_of_two(alignment) ||
      value > std::numeric_limits<std::uint64_t>::max() - (alignment - 1U))
    return std::nullopt;
  return (value + alignment - 1U) & ~(alignment - 1U);
}

bool overlaps(std::uint64_t lhs_offset, std::uint64_t lhs_size, std::uint64_t rhs_offset,
              std::uint64_t rhs_size) {
  return lhs_offset < rhs_offset + rhs_size && rhs_offset < lhs_offset + lhs_size;
}

constexpr std::array policies{
    client::AllocationPolicy::RequireCma, client::AllocationPolicy::RequireDms,
    client::AllocationPolicy::PreferDmsThenCma, client::AllocationPolicy::RequireHostCpu};

struct Placement {
  CarrierRequirement requirement;
  std::uint64_t offset = 0;
};

bool place_group(const std::vector<std::size_t>& members, std::vector<Placement>& carriers,
                 std::uint64_t alignment, const std::vector<std::size_t>& placed,
                 const CarrierLifetimeConflict& conflicts, std::string* error) {
  std::uint64_t cursor = 0;
  while (true) {
    const auto base = align_up(cursor, alignment);
    if (!base)
      return fail(error, "carrier placement overflows");
    bool conflict = false;
    std::uint64_t next = cursor;
    for (const auto index : members) {
      const auto& member = carriers[index];
      std::uint64_t offset;
      if (!add(*base, member.offset, offset))
        return fail(error, "carrier group placement overflows");
      for (const auto other_index : placed) {
        const auto& other = carriers[other_index];
        if (member.requirement.policy != other.requirement.policy ||
            !conflicts(member.requirement, other.requirement) ||
            !overlaps(offset, member.requirement.size_bytes, other.offset,
                      other.requirement.size_bytes))
          continue;
        std::uint64_t end;
        if (!add(other.offset, other.requirement.size_bytes, end))
          return fail(error, "placed carrier extent overflows");
        next = std::max(next, end > member.offset ? end - member.offset : 0U);
        conflict = true;
      }
    }
    if (!conflict) {
      for (const auto index : members) {
        auto& member = carriers[index];
        std::uint64_t end;
        if (!add(*base, member.offset, member.offset) ||
            !add(member.offset, member.requirement.size_bytes, end))
          return fail(error, "carrier group extent overflows");
      }
      return true;
    }
    if (next <= *base)
      return fail(error, "carrier first-fit placement made no progress");
    cursor = next;
  }
}
} // namespace

std::optional<client::AllocationPolicy>
resolve_allocation_policy(std::uint32_t device_access,
                          client::AllocationPolicy eligible_policy) noexcept {
  if (!device_access ||
      std::find(policies.begin(), policies.end(), eligible_policy) == policies.end())
    return std::nullopt;
  return device_access & client::kAccessEv74 ? client::AllocationPolicy::RequireCma
                                             : eligible_policy;
}

const client::FrameSpan* CarrierPacking::span(std::uint64_t carrier_id) const noexcept {
  const auto found = std::find_if(carriers.begin(), carriers.end(),
                                  [&](const auto& item) { return item.carrier_id == carrier_id; });
  return found == carriers.end() ? nullptr : &found->span;
}

std::optional<CarrierPacking> pack_carriers(std::vector<CarrierRequirement> requirements,
                                            StorageReuse reuse,
                                            const CarrierLifetimeConflict& lifetimes_conflict,
                                            std::string* error) {
  if (requirements.empty()) {
    fail(error, "carrier packing requires materialized carriers");
    return std::nullopt;
  }
  std::unordered_set<std::uint64_t> ids;
  for (const auto& requirement : requirements) {
    if (!ids.insert(requirement.carrier_id).second || !requirement.size_bytes ||
        !is_power_of_two(requirement.required_alignment_bytes) ||
        requirement.lifetime.first_sequence > requirement.lifetime.last_sequence ||
        !requirement.required_device_access ||
        std::find(policies.begin(), policies.end(), requirement.policy) == policies.end()) {
      fail(error, "carrier packing has an invalid identity, extent, alignment or policy");
      return std::nullopt;
    }
    if ((requirement.required_device_access & client::kAccessEv74) &&
        requirement.policy != client::AllocationPolicy::RequireCma) {
      fail(error, "EV74 carrier requires CMA storage");
      return std::nullopt;
    }
  }
  std::stable_sort(requirements.begin(), requirements.end(), [](const auto& lhs, const auto& rhs) {
    if (lhs.lifetime.first_sequence != rhs.lifetime.first_sequence)
      return lhs.lifetime.first_sequence < rhs.lifetime.first_sequence;
    return lhs.carrier_id < rhs.carrier_id;
  });
  std::vector<Placement> carriers;
  carriers.reserve(requirements.size());
  for (auto& requirement : requirements)
    carriers.push_back({std::move(requirement), 0});

  if (reuse == StorageReuse::Disabled) {
    for (const auto policy : policies) {
      std::uint64_t cursor = 0;
      for (auto& carrier : carriers) {
        if (carrier.requirement.policy != policy)
          continue;
        const auto offset = align_up(cursor, carrier.requirement.required_alignment_bytes);
        if (!offset || !add(*offset, carrier.requirement.size_bytes, cursor)) {
          fail(error, "carrier no-reuse placement overflows");
          return std::nullopt;
        }
        carrier.offset = *offset;
      }
    }
  } else {
    if (!lifetimes_conflict) {
      fail(error, "carrier reuse requires a lifetime proof");
      return std::nullopt;
    }
    struct Group {
      StorageLifetime lifetime;
      client::AllocationPolicy policy;
      std::vector<std::size_t> members;
      std::uint64_t size = 0;
      std::uint64_t alignment = 0;
    };
    std::vector<Group> groups;
    for (std::size_t index = 0; index < carriers.size(); ++index) {
      const auto& requirement = carriers[index].requirement;
      auto group = std::find_if(groups.begin(), groups.end(), [&](const auto& item) {
        return item.policy == requirement.policy &&
               item.lifetime.first_sequence == requirement.lifetime.first_sequence &&
               item.lifetime.last_sequence == requirement.lifetime.last_sequence;
      });
      if (group == groups.end()) {
        groups.push_back({requirement.lifetime, requirement.policy, {}, 0, 0});
        group = std::prev(groups.end());
      }
      group->members.push_back(index);
    }
    for (auto& group : groups) {
      std::stable_sort(group.members.begin(), group.members.end(), [&](auto lhs, auto rhs) {
        const auto& left = carriers[lhs].requirement;
        const auto& right = carriers[rhs].requirement;
        if (left.required_alignment_bytes != right.required_alignment_bytes)
          return left.required_alignment_bytes > right.required_alignment_bytes;
        if (left.size_bytes != right.size_bytes)
          return left.size_bytes > right.size_bytes;
        return left.carrier_id < right.carrier_id;
      });
      for (const auto index : group.members) {
        auto& carrier = carriers[index];
        const auto offset = align_up(group.size, carrier.requirement.required_alignment_bytes);
        if (!offset || !add(*offset, carrier.requirement.size_bytes, group.size)) {
          fail(error, "carrier lifetime-group extent overflows");
          return std::nullopt;
        }
        carrier.offset = *offset;
        group.alignment = std::max(group.alignment, carrier.requirement.required_alignment_bytes);
      }
    }
    std::stable_sort(groups.begin(), groups.end(), [](const auto& lhs, const auto& rhs) {
      if (lhs.size != rhs.size)
        return lhs.size > rhs.size;
      if (lhs.alignment != rhs.alignment)
        return lhs.alignment > rhs.alignment;
      if (lhs.lifetime.first_sequence != rhs.lifetime.first_sequence)
        return lhs.lifetime.first_sequence < rhs.lifetime.first_sequence;
      return lhs.lifetime.last_sequence < rhs.lifetime.last_sequence;
    });
    std::vector<std::size_t> placed;
    placed.reserve(carriers.size());
    for (const auto& group : groups) {
      if (!place_group(group.members, carriers, group.alignment, placed, lifetimes_conflict, error))
        return std::nullopt;
      placed.insert(placed.end(), group.members.begin(), group.members.end());
    }
  }

  CarrierPacking result;
  for (const auto policy : policies) {
    std::uint64_t bytes = 0;
    std::uint64_t alignment = 0;
    std::uint32_t access = 0;
    for (const auto& carrier : carriers) {
      if (carrier.requirement.policy != policy)
        continue;
      std::uint64_t end;
      if (!add(carrier.offset, carrier.requirement.size_bytes, end)) {
        fail(error, "carrier allocation extent overflows");
        return std::nullopt;
      }
      bytes = std::max(bytes, end);
      alignment = std::max(alignment, carrier.requirement.required_alignment_bytes);
      access |= carrier.requirement.required_device_access;
    }
    if (!bytes)
      continue;
    const auto rounded = align_up(bytes, alignment);
    if (!rounded || *rounded > std::numeric_limits<std::size_t>::max()) {
      fail(error, "carrier allocation rounding overflows");
      return std::nullopt;
    }
    const auto allocation = result.allocations.size();
    result.allocations.push_back({static_cast<std::size_t>(*rounded), access, policy});
    for (const auto& carrier : carriers) {
      if (carrier.requirement.policy == policy)
        result.carriers.push_back({carrier.requirement.carrier_id,
                                   {allocation, static_cast<std::size_t>(carrier.offset),
                                    static_cast<std::size_t>(carrier.requirement.size_bytes)}});
    }
  }
  if (error)
    error->clear();
  return result;
}

} // namespace simaai::neat::pipeline_internal::sima::static_contract
