/**
 * @file
 * @brief Umbrella include for SiMa NEAT's Session and runtime tensor types.
 *
 * Pulls in the Session/Run lifecycle types (Session, Run, SessionOptions, SessionError,
 * SessionReport) along with the tensor surface area applications interact with at
 * runtime (Tensor, TensorCore, TensorSpec, TensorTypes, TensorAdapters,
 * TensorConversion, TensorOpenCV, TessellatedTensor).
 *
 * Include this instead of cherry-picking individual `pipeline` subheaders.
 */
#pragma once

#include "pipeline/SessionOptions.h"
#include "pipeline/Run.h"
#include "pipeline/Session.h"
#include "pipeline/SessionError.h"
#include "pipeline/SessionReport.h"
#include "pipeline/Tensor.h"
#include "pipeline/TensorAdapters.h"
#include "pipeline/TensorConversion.h"
#include "pipeline/TensorCore.h"
#include "pipeline/TensorOpenCV.h"
#include "pipeline/TensorSpec.h"
#include "pipeline/TensorTypes.h"
#include "pipeline/TessellatedTensor.h"
