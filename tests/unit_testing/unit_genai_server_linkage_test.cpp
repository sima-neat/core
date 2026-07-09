#include "genai/GenAIServer.h"
#include "test_main.h"

#include <stdexcept>
#include <string>

RUN_TEST("unit_genai_server_linkage_test", ([] {
           simaai::neat::genai::GenAIServerOptions options;
           options.host = "127.0.0.1";
           options.port = 0;

           try {
             simaai::neat::genai::GenAIServer server(options);
             server.stop();
           } catch (const std::runtime_error& e) {
             const std::string message = e.what();
             if (message.find("NEAT GenAI/LLiMa support is not available in this build") !=
                 std::string::npos) {
               throw std::runtime_error(
                   "C++ GenAIServer is backed by unavailable LLiMa stubs");
             }
             throw;
           }
         }));
