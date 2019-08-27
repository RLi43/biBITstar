/*********************************************************************
* Software License Agreement (BSD License)
*
*  Copyright (c) 2014, University of Toronto
*  All rights reserved.
*
*  Redistribution and use in source and binary forms, with or without
*  modification, are permitted provided that the following conditions
*  are met:
*
*   * Redistributions of source code must retain the above copyright
*     notice, this list of conditions and the following disclaimer.
*   * Redistributions in binary form must reproduce the above
*     copyright notice, this list of conditions and the following
*     disclaimer in the documentation and/or other materials provided
*     with the distribution.
*   * Neither the name of the University of Toronto nor the names of its
*     contributors may be used to endorse or promote products derived
*     from this software without specific prior written permission.
*
*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
*  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
*  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
*  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
*  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
*  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
*  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
*  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
*  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
*  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
*  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
*  POSSIBILITY OF SUCH DAMAGE.
*********************************************************************/

/* Authors: Jonathan Gammell */

// My definition:
#include "../biBITstar.h"

// For stringstreams
#include <sstream>
// For stream manipulations
#include <iomanip>
// For smart pointers
#include <memory>
// For boost::adaptors::reverse which let "for (auto ...)" loops iterate in reverse
#include <boost/range/adaptor/reversed.hpp>

// For OMPL_INFORM et al.
#include "ompl/util/Console.h"
// For exceptions:
#include "ompl/util/Exception.h"
// For toString
#include "ompl/util/String.h"
// For the default optimization objective:
#include "ompl/base/objectives/PathLengthOptimizationObjective.h"

// BIT*:
// A collection of common helper functions
#include "../datastructures/HelperFunctions.h"
// The Vertex ID generator class
#include "../datastructures/IdGenerator.h"
// My vertex class:
#include "../datastructures/Vertex.h"
// My cost & heuristic helper class
#include "../datastructures/CostHelper.h"
// My implicit graph
#include "../datastructures/ImplicitGraph.h"
// My queue class
#include "../datastructures/SearchQueue.h"

#ifdef BIBITSTAR_DEBUG
    #warning Compiling biBIT* with debug-level asserts
#endif  // BIBITSTAR_DEBUG

namespace ompl
{
    namespace geometric
    {
        /////////////////////////////////////////////////////////////////////////////////////////////
        // Public functions:
        biBITstar::biBITstar(const ompl::base::SpaceInformationPtr &si, const std::string &name /*= "biBITstar"*/)
          : ompl::base::Planner(si, name)
        {
#ifdef BIBITSTAR_DEBUG
            OMPL_DEBUG("%s: Compiled with debug-level asserts.", Planner::getName().c_str());
#endif  // BIBITSTAR_DEBUG

            // Allocate my helper classes, they hold settings and must never be deallocated. Give them a pointer to my
            // name, so they can output helpful error messages
            costHelpPtr_ = std::make_shared<CostHelper>();
            graphPtr_ = std::make_shared<ImplicitGraph>([this]()
                                                        {
                                                            return getName();
                                                        });
            queuePtr_ = std::make_shared<SearchQueue>([this]()
                                                      {
                                                          return getName();
                                                      });
            queuePtr_->setPruneDuringResort(usePruning_);

            // Make sure the default name reflects the default k-nearest setting
            if (graphPtr_->getUseKNearest() && Planner::getName() == "biBITstar")
            {
                // It's the current default r-disc BIT* name, but we're using k-nearest, so change
                Planner::setName("kbiBITstar");
            }
            else if (!graphPtr_->getUseKNearest() && Planner::getName() == "kbiBITstar")
            {
                // It's the current default k-nearest BIT* name, but we're using r-disc, so change
                Planner::setName("biBITstar");
            }
            // It's not default named, don't change it

            // Specify my planner specs:
            Planner::specs_.recognizedGoal = ompl::base::GOAL_SAMPLEABLE_REGION;
            Planner::specs_.multithreaded = false;
            // Approximate solutions are supported but must be enabled with the appropriate configuration parameter.
            Planner::specs_.approximateSolutions = graphPtr_->getTrackApproximateSolutions();
            Planner::specs_.optimizingPaths = true;
            Planner::specs_.directed = true;
            Planner::specs_.provingSolutionNonExistence = false;
            Planner::specs_.canReportIntermediateSolutions = true;

            // Register my setting callbacks
            Planner::declareParam<double>("rewire_factor", this, &biBITstar::setRewireFactor, &biBITstar::getRewireFactor,
                                          "1.0:0.01:3.0");
            Planner::declareParam<unsigned int>("samples_per_batch", this, &biBITstar::setSamplesPerBatch,
                                                &biBITstar::getSamplesPerBatch, "1:1:1000000");
            Planner::declareParam<bool>("use_k_nearest", this, &biBITstar::setUseKNearest, &biBITstar::getUseKNearest, "0,"
                                                                                                                   "1");
            Planner::declareParam<bool>("use_graphPtr_pruning", this, &biBITstar::setPruning, &biBITstar::getPruning, "0,"
                                                                                                                  "1");
            Planner::declareParam<double>("prune_threshold_as_fractional_cost_change", this,
                                          &biBITstar::setPruneThresholdFraction, &biBITstar::getPruneThresholdFraction,
                                          "0.0:0.01:1.0");
            Planner::declareParam<bool>("delay_rewiring_to_first_solution", this,
                                        &biBITstar::setDelayRewiringUntilInitialSolution,
                                        &biBITstar::getDelayRewiringUntilInitialSolution, "0,1");
            Planner::declareParam<bool>("use_just_in_time_sampling", this, &biBITstar::setJustInTimeSampling,
                                        &biBITstar::getJustInTimeSampling, "0,1");
            Planner::declareParam<bool>("drop_unconnected_samples_on_prune", this, &biBITstar::setDropSamplesOnPrune,
                                        &biBITstar::getDropSamplesOnPrune, "0,1");
            Planner::declareParam<bool>("stop_on_each_solution_improvement", this, &biBITstar::setStopOnSolnImprovement,
                                        &biBITstar::getStopOnSolnImprovement, "0,1");
            Planner::declareParam<bool>("use_strict_queue_ordering", this, &biBITstar::setStrictQueueOrdering,
                                        &biBITstar::getStrictQueueOrdering, "0,1");
            Planner::declareParam<bool>("find_approximate_solutions", this, &biBITstar::setConsiderApproximateSolutions,
                                        &biBITstar::getConsiderApproximateSolutions, "0,1");

            // Register my progress info:
            addPlannerProgressProperty("best cost DOUBLE", [this]
                                       {
                                           return bestCostProgressProperty();
                                       });
            addPlannerProgressProperty("number of segments in solution path INTEGER", [this]
                                       {
                                           return bestLengthProgressProperty();
                                       });
            addPlannerProgressProperty("current free states INTEGER", [this]
                                       {
                                           return currentFreeProgressProperty();
                                       });
            addPlannerProgressProperty("current graph vertices INTEGER", [this]
                                       {
                                           return currentVertexProgressProperty();
                                       });
            addPlannerProgressProperty("state collision checks INTEGER", [this]
                                       {
                                           return stateCollisionCheckProgressProperty();
                                       });
            addPlannerProgressProperty("edge collision checks INTEGER", [this]
                                       {
                                           return edgeCollisionCheckProgressProperty();
                                       });
            addPlannerProgressProperty("nearest neighbour calls INTEGER", [this]
                                       {
                                           return nearestNeighbourProgressProperty();
                                       });

            // Extra progress info that aren't necessary for every day use. Uncomment if desired.
            /*
            addPlannerProgressProperty("vertex queue size INTEGER", [this]
                                       {
                                           return vertexQueueSizeProgressProperty();
                                       });
            addPlannerProgressProperty("edge queue size INTEGER", [this]
                                       {
                                           return edgeQueueSizeProgressProperty();
                                       });
            addPlannerProgressProperty("iterations INTEGER", [this]
                                       {
                                           return iterationProgressProperty();
                                       });
            addPlannerProgressProperty("batches INTEGER", [this]
                                       {
                                           return batchesProgressProperty();
                                       });
            addPlannerProgressProperty("graph prunings INTEGER", [this]
                                       {
                                           return pruningProgressProperty();
                                       });
            addPlannerProgressProperty("total states generated INTEGER", [this]
                                       {
                                           return totalStatesCreatedProgressProperty();
                                       });
            addPlannerProgressProperty("vertices constructed INTEGER", [this]
                                       {
                                           return verticesConstructedProgressProperty();
                                       });
            addPlannerProgressProperty("states pruned INTEGER", [this]
                                       {
                                           return statesPrunedProgressProperty();
                                       });
            addPlannerProgressProperty("graph vertices disconnected INTEGER", [this]
                                       {
                                           return verticesDisconnectedProgressProperty();
                                       });
            addPlannerProgressProperty("rewiring edges INTEGER", [this]
                                       {
                                           return rewiringProgressProperty();
                                       });
            */
        }

        void biBITstar::setup()
        {
            // Call the base class setup. Marks Planner::setup_ as true.
            Planner::setup();

            // Check if we have a problem definition
            if (static_cast<bool>(Planner::pdef_))
            {
                // We do, do some initialization work.
                // See if we have an optimization objective
                if (!Planner::pdef_->hasOptimizationObjective())
                {
                    /*info */OMPL_DEBUG("%s: No optimization objective specified. Defaulting to optimizing path length.",
                                Planner::getName().c_str());
                    Planner::pdef_->setOptimizationObjective(
                        std::make_shared<base::PathLengthOptimizationObjective>(Planner::si_));
                }
                // No else, we were given one.

                // If the problem definition *has* a goal, make sure it is of appropriate type
                if (static_cast<bool>(Planner::pdef_->getGoal()))
                {
                    if (!Planner::pdef_->getGoal()->hasType(ompl::base::GOAL_SAMPLEABLE_REGION))
                    {
                        OMPL_ERROR("%s::setup() BIT* currently only supports goals that can be cast to a sampleable goal "
                                   "region.",
                                   Planner::getName().c_str());
                        // Mark as not setup:
                        Planner::setup_ = false;
                        return;
                    }
                    // No else, of correct type.
                }
                // No else, called without a goal. Is this MoveIt?

                // Setup the CostHelper, it provides everything I need from optimization objective plus some frills
                costHelpPtr_->setup(Planner::pdef_->getOptimizationObjective(), graphPtr_.get());

                // Setup the queue
                queuePtr_->setup(costHelpPtr_.get(), graphPtr_.get());

                // Setup the graph, it does not hold a copy of this or Planner::pis_, but uses them to create a NN struct
                // and check for starts/goals, respectively.
                graphPtr_->setup(Planner::si_, Planner::pdef_, costHelpPtr_.get(), queuePtr_.get(), this, Planner::pis_);

                // Set the best and pruned costs to the proper objective-based values:
                bestCost_ = costHelpPtr_->infiniteCost();
                prunedCost_ = costHelpPtr_->infiniteCost();

                // Get the measure of the problem
                prunedMeasure_ = Planner::si_->getSpaceMeasure();

                // We are already marked as setup.
            }
            else
            {
                // We don't, so we can't setup. Make sure that is explicit.
                Planner::setup_ = false;
            }
        }

        void biBITstar::clear()
        {
            // Clear all the variables.
            // Keep this in the order of the constructors:

            // The various helper classes. DO NOT reset the pointers, they hold configuration parameters:
            costHelpPtr_->clear();
            graphPtr_->clear();
            queuePtr_->clear();

            // Reset the various calculations and convenience containers. Will be recalculated on setup
            //curGoalVertex_.reset();
            bestCost_ = ompl::base::Cost(std::numeric_limits<double>::infinity());
            bestLength_ = 0u;
            prunedCost_ = ompl::base::Cost(std::numeric_limits<double>::infinity());
            prunedMeasure_ = 0.0;
            hasExactSolution_ = false;
            stopLoop_ = false;
            numBatches_ = 0u;
            numPrunings_ = 0u;
            numIterations_ = 0u;
            numEdgeCollisionChecks_ = 0u;
            numRewirings_ = 0u;

            // DO NOT reset the configuration parameters:
            // samplesPerBatch_
            // usePruning_
            // pruneFraction_
            // stopOnSolnChange_

            // Mark as not setup:
            Planner::setup_ = false;

            // Call my base clear:
            Planner::clear();
        }

        ompl::base::PlannerStatus biBITstar::solve(const ompl::base::PlannerTerminationCondition &ptc)
        {
            // Check that Planner::setup_ is true, if not call this->setup()
            Planner::checkValidity();

            // Assert setup succeeded
            if (!Planner::setup_)
            {
                throw ompl::Exception("%s::solve() failed to set up the planner. Has a problem definition been set?", Planner::getName().c_str());
            }
            // No else

            /*info */OMPL_DEBUG("%s: Searching for a solution to the given planning problem.", Planner::getName().c_str());

            // Reset the manual stop to the iteration loop:
            stopLoop_ = false;

            // If we don't have a goal yet, recall updateStartAndGoalStates, but wait for the first goal (or until the
            // PTC comes true and we give up):
            if (!graphPtr_->hasAGoal())
            {
                OMPL_ERROR("no goal.");
                //graphPtr_->updateStartAndGoalStates(Planner::pis_, ptc);
            }

            // Warn if we are missing a start
            if (!graphPtr_->hasAStart())
            {
                OMPL_ERROR("no start.");
                // We don't have a start, since there's no way to wait for one to appear, so we will not be solving this "problem" today
                OMPL_DEBUG("%s: A solution cannot be found as no valid start states are available.", Planner::getName().c_str());
            }
            // No else, it's a start

            // Warn if we are missing a goal
            if (!graphPtr_->hasAGoal())
            {
                // We don't have a goal (and we waited as long as ptc allowed us for one to appear), so we will not be solving this "problem" today
                OMPL_DEBUG("%s: A solution cannot be found as no valid goal states are available.", Planner::getName().c_str());
            }
            // No else, there's a goal to all of this

            /* Iterate as long as:
              - We're allowed (ptc == false && stopLoop_ == false), AND
              - We haven't found a good enough solution (costHelpPtr_->isSatisfied(bestCost) == false),
              - AND
                - There is a theoretically better solution (costHelpPtr_->isCostBetterThan(graphPtr_->minCost(),
              bestCost_) == true), OR
                - There is are start/goal states we've yet to consider (pis_.haveMoreStartStates() == true ||
              pis_.haveMoreGoalStates() == true)
            */
            while (!ptc && !stopLoop_ && !costHelpPtr_->isSatisfied(bestCost_) &&
                   (costHelpPtr_->isCostBetterThan(graphPtr_->minCost(), bestCost_) ||
                    Planner::pis_.haveMoreStartStates() || Planner::pis_.haveMoreGoalStates()))
            {
                this->iterate();
#ifdef BIBITSTAR_DEBUG
                OMPL_DEBUG("iterate %d", numIterations_);
                OMPL_DEBUG("totally vertex num: %d, H: %d, G: %d; sample num: %d."
                        ,graphPtr_->numConnectedVertices(), graphPtr_->numConnectedVerticesH(), graphPtr_->numConnectedVerticesG(),
                        graphPtr_->numFreeSamples());
                OMPL_DEBUG("queue: edge: %d, vertex: %d",queuePtr_->numEdges(),queuePtr_->numVertices());
                queuePtr_->printInfo();
#endif
            }

            // Announce
            if (hasExactSolution_)
            {
                this->endSuccessMessage();
            }
            else
            {
                this->endFailureMessage();
            }

            // Publish
            if (hasExactSolution_ || graphPtr_->getTrackApproximateSolutions())
            {
                // Any solution
                this->publishSolution();
            }
            // No else, no solution to publish

            // From OMPL's point-of-view, BIT* can always have an approximate solution, so mark solution true if either
            // we have an exact solution or are finding approximate ones.
            // Our returned solution will only be approximate if it is not exact and we are finding approximate
            // solutions.
            // PlannerStatus(addedSolution, approximate)
            return ompl::base::PlannerStatus(hasExactSolution_ || graphPtr_->getTrackApproximateSolutions(),
                                             !hasExactSolution_ && graphPtr_->getTrackApproximateSolutions());
        }

        void biBITstar::getPlannerData(ompl::base::PlannerData &data) const
        {
            // Get the base planner class data:
            Planner::getPlannerData(data);

            // Add the samples (the graph)
            graphPtr_->getGraphAsPlannerData(data);

            // Did we find a solution?
            if (hasExactSolution_)
            {
                // Exact solution
                data.markGoalState(graphPtr_->goalVertex_->stateConst());
            }
            // else if (!hasExactSolution_ && graphPtr_->getTrackApproximateSolutions())
            // {
            //     // Approximate solution
            //     data.markGoalState(graphPtr_->closestVertexToGoal()->stateConst());
            // }
            // No else, no solution
        }

        std::pair<ompl::base::State const *, ompl::base::State const *> biBITstar::getNextEdgeInQueue()
        {
            // Variable:
            // The next edge as a basic pair of states
            std::pair<ompl::base::State const *, ompl::base::State const *> nextEdge;

            if (!queuePtr_->isEmpty())
            {
                // Variable
                // The edge in the front of the queue
                VertexConstPtrPair frontEdge = queuePtr_->frontEdge();

                // The next edge in the queue:
                nextEdge = std::make_pair(frontEdge.first->stateConst(), frontEdge.second->stateConst());
            }
            else
            {
                // An empty edge:
                nextEdge = std::make_pair<ompl::base::State *, ompl::base::State *>(nullptr, nullptr);
            }

            return nextEdge;
        }

        ompl::base::Cost biBITstar::getNextEdgeValueInQueue()
        {
            // Variable
            // The cost of the next edge
            ompl::base::Cost nextCost;

            if (!queuePtr_->isEmpty())
            {
                // The next cost in the queue:
                nextCost = queuePtr_->frontEdgeValue().at(0u);
            }
            else
            {
                // An infinite cost:
                nextCost = costHelpPtr_->infiniteCost();
            }

            return nextCost;
        }

        void biBITstar::getEdgeQueue(VertexConstPtrPairVector *edgesInQueue)
        {
            queuePtr_->getEdges(edgesInQueue);
        }

        void biBITstar::getVertexQueue(VertexConstPtrVector *verticesInQueue)
        {
            queuePtr_->getVertices(verticesInQueue);
        }

        unsigned int biBITstar::numIterations() const
        {
            return numIterations_;
        }

        ompl::base::Cost biBITstar::bestCost() const
        {
            return bestCost_;
        }

        unsigned int biBITstar::numBatches() const
        {
            return numBatches_;
        }
        /////////////////////////////////////////////////////////////////////////////////////////////

        /////////////////////////////////////////////////////////////////////////////////////////////
        // Protected functions:
        void biBITstar::iterate()
        {
            // Info:
            ++numIterations_;

            // Is the edge queue empty
            if (queuePtr_->isEmpty())
            {
                // Yes, we must have just finished a batch. Increase the resolution of the graph and restart the queue.
                this->newBatch();
            }
            else
            {
                // If the edge queue is not empty, then there is work to do!

                // Variables:
                // The current edge:
                VertexPtrPair bestEdge;

                // Pop the minimum edge
                queuePtr_->popFrontEdge(&bestEdge);
#ifdef BIBITSTAR_DEBUG
                // edge debug:
                OMPL_DEBUG("check top edge...");
                VertexPtr v1 = bestEdge.first;
                VertexPtr v2 = bestEdge.second;
                OMPL_DEBUG("edge debug:(id,Gtree,cost) (%d,%d,%f) -->(%d,%d,%f)",
                            v1->getId(), v1->isGtree(), v1->getCost().value(),
                            v2->getId(), v2->isInTree()?v2->isGtree():-1, v2->isInTree()?v2->getCost().value():-1);
                OMPL_DEBUG("(%f,%f)->(%f,%f)",
                    v1->stateConst()->as<ompl::base::RealVectorStateSpace::StateType>()->values[0],v1->stateConst()->as<ompl::base::RealVectorStateSpace::StateType>()->values[1],
                    v2->stateConst()->as<ompl::base::RealVectorStateSpace::StateType>()->values[0],v2->stateConst()->as<ompl::base::RealVectorStateSpace::StateType>()->values[1]);
#endif
                // In the best case, can this edge improve our solution given the current graph?
                // g_t(v) + c_hat(v,x) + h_hat(x) < g_t(x_g)?
                if (costHelpPtr_->isCostBetterThan(costHelpPtr_->currentHeuristicEdge(bestEdge), bestCost_))
                {
                    bool anotherTree = bestEdge.second->isInTree() && (bestEdge.second->isGtree() != bestEdge.first->isGtree());
                    if(anotherTree){
                        //prune?
                    }
                    else{
                    // What about improving the current graph?
                    // g_t(v) + c_hat(v,x)  < g_t(x)?
                        if (!costHelpPtr_->isCostBetterThan(costHelpPtr_->currentHeuristicToTarget(bestEdge),
                                                        bestEdge.second->getCost()))
                                                        {
#ifdef BIBITSTAR_DEBUG
                                                            OMPL_DEBUG("can't improve graph");
                                                            return;
#endif
                                                        }
                    }
                        // Ok, so it *could* be a useful edge. Do the work of calculating its cost for real

                        // Variables:
                        // The true cost of the edge:
                        ompl::base::Cost trueEdgeCost;

                        // Get the true cost of the edge
                        trueEdgeCost = costHelpPtr_->trueEdgeCost(bestEdge);

                        // Can this actual edge ever improve our solution?
                        // g_hat(v) + c(v,x) + h_hat(x) < g_t(x_g)?
                        if (costHelpPtr_->isCostBetterThan(
                                // costHelpPtr_->combineCosts(costHelpPtr_->costToComeHeuristic(bestEdge.first),
                                //                         trueEdgeCost,
                                //                         costHelpPtr_->costToGoHeuristic(bestEdge.second)),
                                costHelpPtr_->combineCosts(costHelpPtr_->biConnectCostHeuristic(bestEdge),
                                    trueEdgeCost),
                                bestCost_))
                        {
                            // Does this edge have a collision?
                            if (this->checkEdge(bestEdge))
                            {

#ifdef BIBITSTAR_DEBUG
            OMPL_DEBUG("new edge collision free...");
#endif
                                if(anotherTree){

                    // Make sure I am not already connected with it
                    if(bestEdge.first->isConn2Another() && bestEdge.second->isConn2Another()){
                        for (ConnEdge ce : graphPtr_->conn2TreesVector){
                            if(ce.first->getId() == bestEdge.first->getId() && ce.second->getId() == bestEdge.second->getId())
                                return;
                            if(ce.first->getId() == bestEdge.second->getId() && ce.second->getId() == bestEdge.first->getId())
                                return;
                        }
                    }
                    // prune?
                                    this->addEdge(bestEdge,trueEdgeCost);
                                    
                                }
                                // Does the current edge improve our graph?
                                // g_t(v) + c(v,x) < g_t(x)?
                                else if (costHelpPtr_->isCostBetterThan(
                                        costHelpPtr_->combineCosts(bestEdge.first->getCost(), trueEdgeCost),
                                        bestEdge.second->getCost()))
                                {
                                    // YAAAAH. Add the edge! Allowing for the sample to be removed from free if it is
                                    // not currently connected and otherwise propagate cost updates to descendants.
                                    // addEdge will update the queue and handle the extra work that occurs if this edge
                                    // improves the solution.
                                    this->addEdge(bestEdge, trueEdgeCost);

                                    // If the path to the goal has changed, we will need to update the cached info about
                                    // the solution cost or solution length:
                                    // this->updateGoalVertex();

                                    /*
                                    //Remove any unnecessary incoming edges in the edge queue
                                    queuePtr_->removeExtraEdgesTo(bestEdge.second);
                                    */

                                    // We will only prune the whole graph/samples on a new batch.
                                }
                                // No else, this edge may be useful at some later date.
                            }
                            // No else, we failed
                        }
                        // No else, we failed
                }
                else
                {
                    // The edge cannot improve our solution, and therefore neither can any other edge in the queue. Give
                    // up on the batch:
                    queuePtr_->finish();
                }
            }  // Search queue not empty.
        }

        void biBITstar::newBatch()
        {
#ifdef BIBITSTAR_DEBUG
            OMPL_DEBUG("newBatch...");
#endif
            // Info:
            ++numBatches_;

            // Reset the queue:
            queuePtr_->reset();

            // // Do we need to update our starts or goals?
            // if (Planner::pis_.haveMoreStartStates() || Planner::pis_.haveMoreGoalStates())
            // {
            //     // There are new starts/goals to get.
            //     graphPtr_->updateStartAndGoalStates(Planner::pis_, ompl::base::plannerAlwaysTerminatingCondition());
            // }
            // // No else, we have enough of a problem to do some work, and everything's up to date.

            // Prune the graph (if enabled)
            this->prune();

            // Add a new batch of samples
            graphPtr_->addNewSamples(samplesPerBatch_);
        }

        void biBITstar::prune()
        {
            /* Test if we should we do a little tidying up:
              - Is pruning enabled?
              - Do we have an exact solution?
              - Has the solution changed more than the specified pruning threshold?
            */
            if ((usePruning_) && (hasExactSolution_) &&
                (std::abs(costHelpPtr_->fractionalChange(bestCost_, prunedCost_)) > pruneFraction_))
            {
                // Variables:
                // The number of vertices and samples pruned:
                std::pair<unsigned int, unsigned int> numPruned(0u, 0u);
                // The current measure of the problem space:
                double informedMeasure = graphPtr_->getInformedMeasure(bestCost_);
                // The change in the informed measure
                double relativeMeasure = std::abs((informedMeasure - prunedMeasure_) / prunedMeasure_);

                /* Is there good reason to prune?
                  - Is the informed subset measurably less than the total problem domain?
                  - Has its measured changed more than the specified pruning threshold?
                  - If an informed measure is not available, we'll assume yes
                 */
                if ((graphPtr_->hasInformedMeasure() && informedMeasure < Planner::si_->getSpaceMeasure() &&
                     relativeMeasure > pruneFraction_) ||
                    (!graphPtr_->hasInformedMeasure()))
                {
                    /*info */OMPL_DEBUG("%s: Pruning the planning problem from a solution of %.4f to %.4f, changing the "
                                "problem size from %.4f to %.4f.",
                                Planner::getName().c_str(), prunedCost_.value(), bestCost_.value(), prunedMeasure_,
                                informedMeasure);

                    // Increment the pruning counter:
                    ++numPrunings_;

                    // Prune the graph
                    numPruned = numPruned + graphPtr_->prune(informedMeasure);

                    // Prune the search.
                    numPruned = numPruned + queuePtr_->prune(graphPtr_->goalVertex_);
                    numPruned = numPruned + queuePtr_->prune(graphPtr_->startVertex_);

                    // Store the cost at which we pruned:
                    prunedCost_ = bestCost_;

                    // And the measure:
                    prunedMeasure_ = informedMeasure;

                    /*info */OMPL_DEBUG("%s: Pruning disconnected %d vertices from the tree and completely removed %d samples.",
                                Planner::getName().c_str(), numPruned.first, numPruned.second);
                }
                // No else, it's not worth the work to prune...
            }
            // No else, why was I called?
        }

        void biBITstar::publishSolution()
        {
            //TODO
            // Variable
            // The reverse path of state pointers
            std::vector<const ompl::base::State *> reversePath;
            // Allocate a path geometric
            auto pathGeoPtr = bestPath();

            // Now create the solution
            ompl::base::PlannerSolution soln(pathGeoPtr);

            // Mark the name:
            soln.setPlannerName(Planner::getName());

            // // Mark as approximate if not exact:
            // if (!hasExactSolution_ && graphPtr_->getTrackApproximateSolutions())
            // {
            //     soln.setApproximate(graphPtr_->smallestDistanceToGoal());
            // }

            // Mark whether the solution met the optimization objective:
            soln.optimized_ = costHelpPtr_->isSatisfied(bestCost_);

            // Add the solution to the Problem Definition:
            Planner::pdef_->addSolutionPath(soln);
        }

        /* std::vector<const ompl::base::State *> biBITstar::bestPathFromGoalToStart() const
        {
            // Variables:
            // A vector of states from goal->start:
            std::vector<const ompl::base::State *> reversePath;
            // The vertex used to ascend up from the goal:
            VertexConstPtr curVertex;

            // Iterate up the chain from the goal, creating a backwards vector:
            if (hasExactSolution_)
            {
                // Start at vertex in the goal
                curVertex = curGoalVertex_;
            }
            else if (!hasExactSolution_ && graphPtr_->getTrackApproximateSolutions())
            {
                // Start at the vertex closest to the goal
                curVertex = graphPtr_->closestVertexToGoal();
            }
            else
            {
                throw ompl::Exception("bestPathFromGoalToStart called without an exact or approximate solution.");
            }

            // Insert the goal into the path
            reversePath.push_back(curVertex->stateConst());

            // Then, use the vertex pointer like an iterator. Starting at the goal, we iterate up the chain pushing the
            // *parent* of the iterator into the vector until the vertex has no parent.
            // This will allows us to add the start (as the parent of the first child) and then stop when we get to the
            // start itself, avoiding trying to find its nonexistent child
            for (; !curVertex->isRoot();//Already allocated & initialized
                 curVertex = curVertex->getParentConst())
            {
#ifdef BIBITSTAR_DEBUG
                // Check the case where the chain ends incorrectly.
                if (curVertex->hasParent() == false)
                {
                    throw ompl::Exception("The path to the goal does not originate at a start state. Something went "
                                          "wrong.");
                }
#endif  // BIBITSTAR_DEBUG

                // Push back the parent into the vector as a state pointer:
                reversePath.push_back(curVertex->getParentConst()->stateConst());
            }
            return reversePath;
        }*/

        std::shared_ptr<ompl::geometric::PathGeometric> biBITstar::bestPath() const{

            auto pathGeoPtr = std::make_shared<ompl::geometric::PathGeometric>(Planner::si_);

            std::vector<const ompl::base::State *> reversePath;
            // The vertex used to ascend up from the goal:
            VertexConstPtr curVertex;
            VertexConstPtr curVertex2;

            // Iterate up the chain from the goal, creating a backwards vector:
            if (hasExactSolution_)
            {
                // Start at vertex in the goal
                if(curBestConn_.first->isGtree()){
                    curVertex = curBestConn_.first;
                    curVertex2 = curBestConn_.second;
                    }
                else {
                    curVertex = curBestConn_.second;
                    curVertex2 = curBestConn_.first;
                    }
            }
            else
            {
                throw ompl::Exception("bestPath called without an exact solution.");
            }

            // Insert the conn in gtree into the path
            reversePath.push_back(curVertex->stateConst());
            // Then, use the vertex pointer like an iterator. Starting at the goal, we iterate up the chain pushing the
            // *parent* of the iterator into the vector until the vertex has no parent.
            // This will allows us to add the start (as the parent of the first child) and then stop when we get to the
            // start itself, avoiding trying to find its nonexistent child
            for (; !curVertex->isRoot();//Already allocated & initialized
                 curVertex = curVertex->getParentConst())
            {
#ifdef BIBITSTAR_DEBUG
                // Check the case where the chain ends incorrectly.
                if (curVertex->hasParent() == false)
                {
                    throw ompl::Exception("The path to the goal does not originate at a start state. Something went "
                                          "wrong.");
                }
#endif  // BIBITSTAR_DEBUG

                // Push back the parent into the vector as a state pointer:
                reversePath.push_back(curVertex->getParentConst()->stateConst());
            }

            // Now iterate that vector in reverse, putting the states into the path geometric
            for (const auto &solnState : boost::adaptors::reverse(reversePath))
            {
                pathGeoPtr->append(solnState);
            }

            pathGeoPtr->append(curVertex2->stateConst());
            for (; !curVertex2->isRoot();//Already allocated & initialized
                 curVertex2 = curVertex2->getParentConst())
            {
#ifdef BIBITSTAR_DEBUG
                // Check the case where the chain ends incorrectly.
                if (curVertex->hasParent() == false)
                {
                    throw ompl::Exception("The path to the goal does not originate at a start state. Something went "
                                          "wrong.");
                }
#endif  // BIBITSTAR_DEBUG

                // Push back the parent into the vector as a state pointer:
                pathGeoPtr->append(curVertex2->getParentConst()->stateConst());
            }
            
            pathGeoPtr->append(curVertex2->stateConst());

            return pathGeoPtr;
        }

        bool biBITstar::checkEdge(const VertexConstPtrPair &edge)
        {
            ++numEdgeCollisionChecks_;
            return Planner::si_->checkMotion(edge.first->stateConst(), edge.second->stateConst());
        }

        void biBITstar::addEdge(const VertexPtrPair &newEdge, const ompl::base::Cost &edgeCost)
        {
#ifdef BIBITSTAR_DEBUG
            if (newEdge.first->isInTree() == false)
            {
                throw ompl::Exception("Adding an edge from a vertex not connected to the graph");
            }
            if (costHelpPtr_->isCostEquivalentTo(costHelpPtr_->trueEdgeCost(newEdge), edgeCost) == false)
            {
                throw ompl::Exception("You have passed the wrong edge cost to addEdge.");
            }
#endif  // BIBITSTAR_DEBUG

            if(newEdge.second->isInTree()){
                if(newEdge.first->isGtree() == newEdge.second->isGtree()){
                    // rewiring

#ifdef BIBITSTAR_DEBUG
                    OMPL_DEBUG("rewire");
#endif
                    this->replaceParent(newEdge, edgeCost);
                }
                else{
                    // new solution
                    // Mark that we have a solution

#ifdef BIBITSTAR_DEBUG
                    OMPL_DEBUG("new solution.");
#endif
                    hasExactSolution_ = true;
                    graphPtr_->conn2TreesVector.push_back(newEdge);


                    ompl::base::Cost newCost = costHelpPtr_->combineCosts(newEdge.first->getCost(),
                                                                        newEdge.second->getCost(),
                                                                        edgeCost);
                    // if a better solution
                    if (costHelpPtr_->isCostBetterThan(newCost,bestCost_)){
                        bestCost_ = newCost;
                        curBestConn_ = newEdge;
                        // and best length
                        bestLength_ = curBestConn_.first->getDepth() + curBestConn_.second->getDepth() + 1u;
                        // Tell everyone else about it.
                        queuePtr_->hasSolution(bestCost_);
                        graphPtr_->hasSolution(bestCost_);

                        // Stop the solution loop if enabled:
                        stopLoop_ = stopOnSolnChange_;

                        // Brag:
                        this->goalMessage();

                        // // If enabled, pass the intermediate solution back through the call back:
                        // if (static_cast<bool>(Planner::pdef_->getIntermediateSolutionCallback()))
                        // {
                        //     // The form of path passed to the intermediate solution callback is not well documented, but it
                        //     // *appears* that it's not supposed
                        //     // to include the start or goal; however, that makes no sense for multiple start/goal problems, so
                        //     // we're going to include it anyway (sorry).
                        //     // Similarly, it appears to be ordered as (goal, goal-1, goal-2,...start+1, start) which
                        //     // conveniently allows us to reuse code.
                        //     Planner::pdef_->getIntermediateSolutionCallback()(this, this->bestPathFromGoalToStart(), bestCost_);
                        // }
                    }
                }

            }
            else{
                // to free sample, we just add the vertex
#ifdef BIBITSTAR_DEBUG
                OMPL_DEBUG("free sample add in");
                graphPtr_->assertValidSample(newEdge.second, false);
#endif  // BIBITSTAR_DEBUG
                // Add a parent to the child, updating descendant costs:
                newEdge.second->addParent(newEdge.first, edgeCost, true);
                // Add a child to the parent:
                newEdge.first->addChild(newEdge.second);
                // Then enqueue the vertex, moving it from the free set to the vertex set.
                queuePtr_->enqueueVertex(newEdge.second, true, newEdge.first->isGtree());
            }
        }

        void biBITstar::replaceParent(const VertexPtrPair &newEdge, const ompl::base::Cost &edgeCost)
        {
#ifdef BIBITSTAR_DEBUG
            if (newEdge.second->getParent()->getId() == newEdge.first->getId())
            {
                throw ompl::Exception("The new and old parents of the given rewiring are the same.");
            }
            if (costHelpPtr_->isCostBetterThan(newEdge.second->getCost(),
                                               costHelpPtr_->combineCosts(newEdge.first->getCost(), edgeCost)) == true)
            {
                throw ompl::Exception("The new edge will increase the cost-to-come of the vertex!");
            }
#endif  // BIBITSTAR_DEBUG

            // Increment our counter:
            ++numRewirings_;

            // Remove the child from the parent, not updating costs
            newEdge.second->getParent()->removeChild(newEdge.second);

            // Remove the parent from the child, not updating costs
            newEdge.second->removeParent(false);

            // Add the parent to the child. updating the downstream costs.
            newEdge.second->addParent(newEdge.first, edgeCost, true);

            // Add the child to the parent.
            newEdge.first->addChild(newEdge.second);

            // Mark the queues as unsorted below this child
            queuePtr_->markVertexUnsorted(newEdge.second);
        }

        /* void biBITstar::updateGoalVertex()
        {
            // Variable
            // Whether we've updated the goal, be pessimistic.
            bool goalUpdated = false;
            // The new goal, start with the current goal
            VertexConstPtr newBestGoal = curGoalVertex_;
            // The new cost, start as the current bestCost_
            ompl::base::Cost newCost = bestCost_;

            // Iterate through the vector of goals, and see if the solution has changed
            for (auto goalIter = graphPtr_->goalVerticesBeginConst(); goalIter != graphPtr_->goalVerticesEndConst();
                 ++goalIter)
            {
                // First, is this goal even in the tree?
                if ((*goalIter)->isInTree())
                {
                    // Next, is there currently a solution?
                    if (static_cast<bool>(newBestGoal))
                    {
                        // There is already a solution, is it to this goal?
                        if ((*goalIter)->getId() == newBestGoal->getId())
                        {
                            // Ah-ha, We meet again! Are we doing any better? We check the length as sometimes the path
                            // length changes with minimal change in cost.
                            if (!costHelpPtr_->isCostEquivalentTo((*goalIter)->getCost(), newCost) ||
                                ((*goalIter)->getDepth() + 1u) != bestLength_)
                            {
                                // The path to the current best goal has changed, so we need to update it.
                                goalUpdated = true;
                                newBestGoal = *goalIter;
                                newCost = newBestGoal->getCost();
                            }
                            // No else, no change
                        }
                        else
                        {
                            // It is not to this goal, we have a second solution! What an easy problem... but is it
                            // better?
                            if (costHelpPtr_->isCostBetterThan((*goalIter)->getCost(), newCost))
                            {
                                // It is! Save this as a better goal:
                                goalUpdated = true;
                                newBestGoal = *goalIter;
                                newCost = newBestGoal->getCost();
                            }
                            // No else, not a better solution
                        }
                    }
                    else
                    {
                        // There isn't a preexisting solution, that means that any goal is an update:
                        goalUpdated = true;
                        newBestGoal = *goalIter;
                        newCost = newBestGoal->getCost();
                    }
                }
                // No else, can't be a better solution if it's not in the spanning tree, can it?
            }

            // Did we update the goal?
            if (goalUpdated)
            {
                // Mark that we have a solution
                hasExactSolution_ = true;

                // Store the current goal
                curGoalVertex_ = newBestGoal;

                // Update the best cost:
                bestCost_ = newCost;

                // and best length
                bestLength_ = curGoalVertex_->getDepth() + 1u;

                // Tell everyone else about it.
                queuePtr_->hasSolution(bestCost_);
                graphPtr_->hasSolution(bestCost_);

                // Stop the solution loop if enabled:
                stopLoop_ = stopOnSolnChange_;

                // Brag:
                this->goalMessage();

                // If enabled, pass the intermediate solution back through the call back:
                if (static_cast<bool>(Planner::pdef_->getIntermediateSolutionCallback()))
                {
                    // The form of path passed to the intermediate solution callback is not well documented, but it
                    // *appears* that it's not supposed
                    // to include the start or goal; however, that makes no sense for multiple start/goal problems, so
                    // we're going to include it anyway (sorry).
                    // Similarly, it appears to be ordered as (goal, goal-1, goal-2,...start+1, start) which
                    // conveniently allows us to reuse code.
                    Planner::pdef_->getIntermediateSolutionCallback()(this, this->bestPathFromGoalToStart(), bestCost_);
                }
            }
            // No else, the goal didn't change
        }*/

        void biBITstar::goalMessage() const
        {
            /*info */OMPL_DEBUG("%s (%u iters): Found a solution of cost %.4f (%u vertices) from %u samples by processing %u "
                        "edges (%u collision checked) to create %u vertices and perform %u rewirings. The graph "
                        "currently has %u vertices.",
                        Planner::getName().c_str(), numIterations_, bestCost_, bestLength_,
                        graphPtr_->numStatesGenerated(), queuePtr_->numEdgesPopped(), numEdgeCollisionChecks_,
                        graphPtr_->numVerticesConnected(), numRewirings_, graphPtr_->numConnectedVertices());
        }

        void biBITstar::endSuccessMessage() const
        {
            /*info */OMPL_DEBUG("%s: Finished with a solution of cost %.4f (%u vertices) found from %u samples by processing "
                        "%u edges (%u collision checked) to create %u vertices and perform %u rewirings. The final "
                        "graph has %u vertices.",
                        Planner::getName().c_str(), bestCost_, bestLength_, graphPtr_->numStatesGenerated(),
                        queuePtr_->numEdgesPopped(), numEdgeCollisionChecks_, graphPtr_->numVerticesConnected(),
                        numRewirings_, graphPtr_->numConnectedVertices());
        }

        void biBITstar::endFailureMessage() const
        {
            if (graphPtr_->getTrackApproximateSolutions())
            {
                /*info */OMPL_DEBUG("%s (%u iters): Did not find an exact solution from %u samples after processing %u edges "
                            "(%u collision checked) to create %u vertices and perform %u rewirings. The final graph "
                            "has %u vertices. The best approximate solution was %.4f from the goal and has a cost of "
                            "%.4f.",
                            Planner::getName().c_str(), numIterations_, graphPtr_->numStatesGenerated(),
                            queuePtr_->numEdgesPopped(), numEdgeCollisionChecks_, graphPtr_->numVerticesConnected(),
                            numRewirings_, graphPtr_->numConnectedVertices(), graphPtr_->smallestDistanceToGoal(),
                            graphPtr_->closestVertexToGoal()->getCost().value());
            }
            else
            {
                /*info */OMPL_DEBUG("%s (%u iters): Did not find an exact solution from %u samples after processing %u edges "
                            "(%u collision checked) to create %u vertices and perform %u rewirings. The final graph "
                            "has %u vertices.",
                            Planner::getName().c_str(), numIterations_, graphPtr_->numStatesGenerated(),
                            queuePtr_->numEdgesPopped(), numEdgeCollisionChecks_, graphPtr_->numVerticesConnected(),
                            numRewirings_, graphPtr_->numConnectedVertices());
            }
        }

        void biBITstar::statusMessage(const ompl::msg::LogLevel &msgLevel, const std::string &status) const
        {
            // Check if we need to create the message
            if (msgLevel >= ompl::msg::getLogLevel())
            {
                // Variable
                // The message as a stream:
                std::stringstream outputStream;

                // Create the stream:
                // The name of the planner
                outputStream << Planner::getName();
                outputStream << " (";
                // The current path cost:
                outputStream << "l: " << std::setw(6) << std::setfill(' ') << std::setprecision(5) << bestCost_.value();
                // The number of batches:
                outputStream << ", b: " << std::setw(5) << std::setfill(' ') << numBatches_;
                // The number of iterations
                outputStream << ", i: " << std::setw(5) << std::setfill(' ') << numIterations_;
                // The number of states current in the graph
                outputStream << ", g: " << std::setw(5) << std::setfill(' ') << graphPtr_->numConnectedVertices();
                // The number of free states
                outputStream << ", f: " << std::setw(5) << std::setfill(' ') << graphPtr_->numFreeSamples();
                // The number edges in the queue:
                outputStream << ", q: " << std::setw(5) << std::setfill(' ') << queuePtr_->numEdges();
                // The total number of edges taken out of the queue:
                outputStream << ", t: " << std::setw(5) << std::setfill(' ') << queuePtr_->numEdgesPopped();
                // The number of samples generated
                outputStream << ", s: " << std::setw(5) << std::setfill(' ') << graphPtr_->numStatesGenerated();
                // The number of vertices ever added to the graph:
                outputStream << ", v: " << std::setw(5) << std::setfill(' ') << graphPtr_->numVerticesConnected();
                // The number of prunings:
                outputStream << ", p: " << std::setw(5) << std::setfill(' ') << numPrunings_;
                // The number of rewirings:
                outputStream << ", r: " << std::setw(5) << std::setfill(' ') << numRewirings_;
                // The number of nearest-neighbour calls
                outputStream << ", n: " << std::setw(5) << std::setfill(' ') << graphPtr_->numNearestLookups();
                // The number of state collision checks:
                outputStream << ", c(s): " << std::setw(5) << std::setfill(' ') << graphPtr_->numStateCollisionChecks();
                // The number of edge collision checks:
                outputStream << ", c(e): " << std::setw(5) << std::setfill(' ') << numEdgeCollisionChecks_;
                outputStream << "):    ";
                // The message:
                outputStream << status;

                if (msgLevel == ompl::msg::LOG_DEBUG)
                {
                    OMPL_DEBUG("%s: ", outputStream.str().c_str());
                }
                else if (msgLevel == ompl::msg::LOG_INFO)
                {
                    /*info */OMPL_DEBUG("%s: ", outputStream.str().c_str());
                }
                else if (msgLevel == ompl::msg::LOG_WARN)
                {
                    OMPL_DEBUG("%s: ", outputStream.str().c_str());
                }
                else if (msgLevel == ompl::msg::LOG_ERROR)
                {
                    OMPL_ERROR("%s: ", outputStream.str().c_str());
                }
                else
                {
                    throw ompl::Exception("Log level not recognized");
                }
            }
            // No else, this message is below the log level
        }
        /////////////////////////////////////////////////////////////////////////////////////////////

        /////////////////////////////////////////////////////////////////////////////////////////////
        // Boring sets/gets (Public) and progress properties (Protected):
        void biBITstar::setRewireFactor(double rewireFactor)
        {
            graphPtr_->setRewireFactor(rewireFactor);
        }

        double biBITstar::getRewireFactor() const
        {
            return graphPtr_->getRewireFactor();
        }

        void biBITstar::setSamplesPerBatch(unsigned int n)
        {
            samplesPerBatch_ = n;
        }

        unsigned int biBITstar::getSamplesPerBatch() const
        {
            return samplesPerBatch_;
        }

        void biBITstar::setUseKNearest(bool useKNearest)
        {
            // Store
            graphPtr_->setUseKNearest(useKNearest);

            // If the planner is default named, we change it:
            if (!graphPtr_->getUseKNearest() && Planner::getName() == "kbiBITstar")
            {
                // It's current the default k-nearest BIT* name, and we're toggling, so set to the default r-disc
                Planner::setName("biBITstar");
            }
            else if (graphPtr_->getUseKNearest() && Planner::getName() == "biBITstar")
            {
                // It's current the default r-disc BIT* name, and we're toggling, so set to the default k-nearest
                Planner::setName("kbiBITstar");
            }
            // It's not default named, don't change it
        }

        bool biBITstar::getUseKNearest() const
        {
            return graphPtr_->getUseKNearest();
        }

        void biBITstar::setStrictQueueOrdering(bool beStrict)
        {
            queuePtr_->setStrictQueueOrdering(beStrict);
        }

        bool biBITstar::getStrictQueueOrdering() const
        {
            return queuePtr_->getStrictQueueOrdering();
        }

        void biBITstar::setPruning(bool prune)
        {
            if (!prune)
            {
                OMPL_DEBUG("%s: Turning pruning off has never really been tested.", Planner::getName().c_str());
            }

            // Store
            usePruning_ = prune;

            // Pass the setting onto the queue
            queuePtr_->setPruneDuringResort(usePruning_);
        }

        bool biBITstar::getPruning() const
        {
            return usePruning_;
        }

        void biBITstar::setPruneThresholdFraction(double fractionalChange)
        {
            if (fractionalChange < 0.0 || fractionalChange > 1.0)
            {
                throw ompl::Exception("Prune threshold must be specified as a fraction between [0, 1].");
            }

            pruneFraction_ = fractionalChange;
        }

        double biBITstar::getPruneThresholdFraction() const
        {
            return pruneFraction_;
        }

        void biBITstar::setDelayRewiringUntilInitialSolution(bool delayRewiring)
        {
            queuePtr_->setDelayedRewiring(delayRewiring);
        }

        bool biBITstar::getDelayRewiringUntilInitialSolution() const
        {
            return queuePtr_->getDelayedRewiring();
        }

        void biBITstar::setJustInTimeSampling(bool useJit)
        {
            graphPtr_->setJustInTimeSampling(useJit);
        }

        bool biBITstar::getJustInTimeSampling() const
        {
            return graphPtr_->getJustInTimeSampling();
        }

        void biBITstar::setDropSamplesOnPrune(bool dropSamples)
        {
            graphPtr_->setDropSamplesOnPrune(dropSamples);
        }

        bool biBITstar::getDropSamplesOnPrune() const
        {
            return graphPtr_->getDropSamplesOnPrune();
        }

        void biBITstar::setStopOnSolnImprovement(bool stopOnChange)
        {
            stopOnSolnChange_ = stopOnChange;
        }

        bool biBITstar::getStopOnSolnImprovement() const
        {
            return stopOnSolnChange_;
        }

        void biBITstar::setConsiderApproximateSolutions(bool findApproximate)
        {
            // Store
            graphPtr_->setTrackApproximateSolutions(findApproximate);

            // Mark the Planner spec:
            Planner::specs_.approximateSolutions = graphPtr_->getTrackApproximateSolutions();
        }

        bool biBITstar::getConsiderApproximateSolutions() const
        {
            return graphPtr_->getTrackApproximateSolutions();
        }

        template <template <typename T> class NN>
        void biBITstar::setNearestNeighbors()
        {
            // Check if the problem is already setup, if so, the NN structs have data in them and you can't really
            // change them:
            if (Planner::setup_)
            {
                OMPL_DEBUG("%s: The nearest neighbour datastructures cannot be changed once the planner is setup. "
                          "Continuing to use the existing containers.",
                          Planner::getName().c_str());
            }
            else
            {
                graphPtr_->setNearestNeighbors<NN>();
            }
        }

        std::string biBITstar::bestCostProgressProperty() const
        {
            return ompl::toString(this->bestCost().value());
        }

        std::string biBITstar::bestLengthProgressProperty() const
        {
            return std::to_string(bestLength_);
        }

        std::string biBITstar::currentFreeProgressProperty() const
        {
            return std::to_string(graphPtr_->numFreeSamples());
        }

        std::string biBITstar::currentVertexProgressProperty() const
        {
            return std::to_string(graphPtr_->numConnectedVertices());
        }

        std::string biBITstar::vertexQueueSizeProgressProperty() const
        {
            return std::to_string(queuePtr_->numVertices());
        }

        std::string biBITstar::edgeQueueSizeProgressProperty() const
        {
            return std::to_string(queuePtr_->numEdges());
        }

        std::string biBITstar::iterationProgressProperty() const
        {
            return std::to_string(this->numIterations());
        }

        std::string biBITstar::batchesProgressProperty() const
        {
            return std::to_string(this->numBatches());
        }

        std::string biBITstar::pruningProgressProperty() const
        {
            return std::to_string(numPrunings_);
        }

        std::string biBITstar::totalStatesCreatedProgressProperty() const
        {
            return std::to_string(graphPtr_->numStatesGenerated());
        }

        std::string biBITstar::verticesConstructedProgressProperty() const
        {
            return std::to_string(graphPtr_->numVerticesConnected());
        }

        std::string biBITstar::statesPrunedProgressProperty() const
        {
            return std::to_string(graphPtr_->numFreeStatesPruned());
        }

        std::string biBITstar::verticesDisconnectedProgressProperty() const
        {
            return std::to_string(graphPtr_->numVerticesDisconnected());
        }

        std::string biBITstar::rewiringProgressProperty() const
        {
            return std::to_string(numRewirings_);
        }

        std::string biBITstar::stateCollisionCheckProgressProperty() const
        {
            return std::to_string(graphPtr_->numStateCollisionChecks());
        }

        std::string biBITstar::edgeCollisionCheckProgressProperty() const
        {
            return std::to_string(numEdgeCollisionChecks_);
        }

        std::string biBITstar::nearestNeighbourProgressProperty() const
        {
            return std::to_string(graphPtr_->numNearestLookups());
        }

        std::string biBITstar::edgesProcessedProgressProperty() const
        {
            return std::to_string(queuePtr_->numEdgesPopped());
        }
        /////////////////////////////////////////////////////////////////////////////////////////////
    }  // geometric
}  // ompl
