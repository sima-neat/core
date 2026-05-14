/**
 * @file
 * @brief Umbrella include for SiMa NEAT's model layer.
 *
 * Pulls in everything needed to load and use a model: the Model class, the MPK loader
 * and manifest types, the MPK pipeline adapter and PipelineSequence, the ModelGroups
 * factory bundle, plus the detection-pipeline helpers (DetectionTypes, EncodedSampleUtil,
 * StageRun, SimaBoxDecode) most model-driven applications also use.
 *
 * Include this instead of cherry-picking individual model/MPK headers.
 */
#pragma once

#include "model/Model.h"
#include "mpk/MpKLoader.h"
#include "mpk/MpKManifest.h"
#include "mpk/MpKPipelineAdapter.h"
#include "mpk/PipelineSequence.h"
#include "nodes/groups/ModelGroups.h"
#include "nodes/sima/SimaBoxDecode.h"
#include "pipeline/DetectionTypes.h"
#include "pipeline/EncodedSampleUtil.h"
#include "pipeline/StageRun.h"
