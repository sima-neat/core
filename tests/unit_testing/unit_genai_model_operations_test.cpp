#include "genai/internal/ModelOperations.h"
#include "test_main.h"

#include <chrono>
#include <future>
#include <thread>

RUN_TEST("unit_genai_model_operations_test", ([] {
           using namespace std::chrono_literals;
           simaai::neat::genai::internal::ModelOperations operations;
           std::unique_lock active(operations);
           require(!operations.try_lock(), "cache control acquired an active workspace");

           std::promise<bool> acquired;
           auto result = acquired.get_future();
           std::jthread queued([&](std::stop_token stop) {
             const bool entered = operations.lock(stop);
             if (entered)
               operations.unlock();
             acquired.set_value(entered);
           });
           queued.request_stop();
           const bool cancelled_while_busy = result.wait_for(2s) == std::future_status::ready;
           active.unlock();
           require(cancelled_while_busy && !result.get(),
                   "cancelled request waited for the active model");

           require(operations.try_lock(), "idle cache control could not acquire the workspace");
           operations.unlock();
           require(operations.lock(std::stop_token{}), "later request could not acquire the model");
           operations.unlock();
         }));
