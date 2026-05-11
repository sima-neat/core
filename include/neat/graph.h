/**
 * @file
 * @brief Umbrella include for SiMa NEAT's graph layer.
 *
 * Pulls in everything needed to build, compile, and run a Graph: the Graph class itself,
 * the GraphDsl construction helpers, GraphSession/GraphRun runtime types, the Compiler,
 * the StageExecutor and StreamScheduler, the runtime building blocks (BlockingQueue,
 * StageMailbox, StrictSync), and the canonical Graph node implementations (PipelineNode,
 * StageNode, FanOut, JoinBundle, JoinEncodedWithMeta, LambdaStage, Map, StageModelExecutor,
 * StampFrameId, StreamMetadata, plus the Adapters glue).
 *
 * Include this header instead of cherry-picking individual graph subsystem headers.
 */
#pragma once

#include "graph/Compiler.h"
#include "graph/Graph.h"
#include "graph/GraphDsl.h"
#include "graph/GraphHelpers.h"
#include "graph/GraphMetadata.h"
#include "graph/GraphPrinter.h"
#include "graph/GraphRun.h"
#include "graph/GraphSession.h"
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
