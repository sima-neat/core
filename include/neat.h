/**
 * @file
 * @brief Top-level umbrella header for the SiMa NEAT framework.
 *
 * Including `<neat.h>` is enough to write a complete NEAT application: it transitively
 * pulls in the public Graph/runtime types (`neat/runtime.h`), the model layer
 * (`neat/models.h`), the reusable Graph fragment factories (`neat/node_groups.h`),
 * the atomic Node types (`neat/nodes.h`), and the GenAI/LLiMa surface
 * (`neat/genai.h`).
 *
 * Application code should prefer this single include over cherry-picking individual
 * subsystem headers, both for readability and so the framework can keep the public
 * surface area cohesive.
 */
#pragma once

#include "neat/genai.h"
#include "neat/models.h"
#include "neat/node_groups.h"
#include "neat/nodes.h"
#include "neat/runtime.h"
