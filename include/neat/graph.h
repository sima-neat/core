/**
 * @file
 * @brief Advanced/internal umbrella include for SiMa NEAT's lower-level runtime graph layer.
 *
 * Normal application composition should use `simaai::neat::Graph` from `<neat.h>` /
 * `pipeline/Graph.h`. This header exposes the lower-level runtime substrate used by the
 * public Graph compiler: graph::build/GraphRun, StageExecutor, scheduler/mailbox helpers,
 * and internal runtime nodes.
 *
 * Include this header only for runtime/compiler internals or focused internal tests.
 */
#pragma once

#include "graph/Compiler.h"
#include "graph/Graph.h"
#include "graph/GraphDsl.h"
#include "graph/GraphHelpers.h"
#include "graph/GraphMetadata.h"
#include "graph/GraphPrinter.h"
#include "graph/GraphRun.h"
#include "graph/GraphBuild.h"
#include "graph/GraphTypes.h"
#include "graph/Node.h"
#include "graph/StageExecutor.h"
#include "graph/StrictSync.h"
#include "graph/nodes/Adapters.h"
#include "graph/nodes/FanOut.h"
#include "graph/nodes/JoinBundle.h"
#include "graph/nodes/JoinEncodedWithMeta.h"
#include "graph/nodes/LambdaStage.h"
#include "graph/nodes/Map.h"
#include "graph/nodes/PipelineNode.h"
#include "graph/nodes/StageModelExecutor.h"
#include "graph/nodes/StageNode.h"
#include "graph/nodes/StampFrameId.h"
#include "graph/nodes/StreamMetadata.h"
#include "graph/nodes/StreamScheduler.h"
#include "graph/runtime/BlockingQueue.h"
#include "graph/runtime/StageMailbox.h"
