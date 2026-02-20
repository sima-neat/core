#include "pipeline/TensorConversion.h"
#include "test_main.h"
#include "test_utils.h"

RUN_TEST("unit_tensor_conversion_policy_test", ([] {
           using namespace simaai::neat;

           const ConversionCost reinterpret_cost =
               estimate_conversion_cost(ConversionKind::Reinterpret, 128);
           require(reinterpret_cost.compute_class == 0, "Reinterpret compute class should be 0");
           require(reinterpret_cost.bytes_copied == 0,
                   "Reinterpret bytes_copied should be forced to 0");

           const ConversionCost view_cost = estimate_conversion_cost(ConversionKind::View, 256);
           require(view_cost.compute_class == 0, "View compute class should be 0");
           require(view_cost.bytes_copied == 0, "View bytes_copied should be forced to 0");

           const ConversionCost pack_cost = estimate_conversion_cost(ConversionKind::Pack, 512);
           require(pack_cost.compute_class == 0, "Pack compute class should be 0");
           require(pack_cost.bytes_copied == 512, "Pack bytes_copied should preserve input");

           const ConversionCost convert_cost =
               estimate_conversion_cost(ConversionKind::Convert, 1024);
           require(convert_cost.compute_class == 1, "Convert compute class should be 1");

           const ConversionCost transfer_cost =
               estimate_conversion_cost(ConversionKind::Transfer, 2048);
           require(transfer_cost.compute_class == 2, "Transfer compute class should be 2");

           require(!conversion_allowed(ConversionPolicy::Strict, ConversionKind::Reinterpret),
                   "Strict policy should reject conversions");
           require(!conversion_allowed(ConversionPolicy::Strict, ConversionKind::Transfer),
                   "Strict policy should reject transfer conversion");

           require(conversion_allowed(ConversionPolicy::AllowWithTrace, ConversionKind::Convert),
                   "AllowWithTrace should allow conversions");
           require(conversion_allowed(ConversionPolicy::AllowSilent, ConversionKind::Transfer),
                   "AllowSilent should allow conversions");
         }));
