#ifndef FEDERATED1_SPARSIFICATION_H
#define FEDERATED1_SPARSIFICATION_H

#include "federated-types.h"

#include <cstddef>
#include <vector>

namespace federated {

/* Validate the simulation configuration and client inputs. */
void ValidateFederatedRoundInputs(
    const FederatedConfig& config,
    const std::vector<ClientRoundInput>& clients);

/* Chunk score is sum_j update_l[j]^2; use existing entries in a short final chunk, equivalently zero-padding it. */
std::vector<double> ComputeChunkScores(
    const std::vector<double>& update,
    size_t chunkSize);

/* Select highest scores, ordered by decreasing score and then increasing chunk ID. */
std::vector<size_t> SelectTopChunks(
    const std::vector<double>& chunkScores,
    size_t selectedChunkCount);

/* Compute selection, client packing, cohorts, admission and server packing once to produce a fixed plan. */
SparseRoundPlan BuildSparseRoundPlan(
    const FederatedConfig& config,
    const std::vector<ClientRoundInput>& clients);

/* Validate selection/packing agreement, membership, admission, accepted IDs, server groups and plan hash without recomputing scores. */
void ValidateSparseRoundPlan(
    const FederatedConfig& config,
    const std::vector<ClientRoundInput>& clients,
    const SparseRoundPlan& plan);

/* Find the fixed plan for clientId; throw if absent. */
const ClientSparsePlan& FindClientSparsePlan(
    const SparseRoundPlan& plan,
    size_t clientId);

/* Find the cohort for chunkId; throw if absent. */
const ChunkCohort& FindChunkCohort(
    const SparseRoundPlan& plan,
    size_t chunkId);

}  // namespace federated

#endif  // FEDERATED1_SPARSIFICATION_H