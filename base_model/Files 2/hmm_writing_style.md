# Hidden Markov Models Writing Style Guide

## Introduction & Background

\"Although initially introduced and studied in the late 1960s and early 1970s, statistical methods of Markov source or hidden Markov modeling have become increasingly popular in the last several years.\"

\"There are two strong reasons why this has occurred. First the models are very rich in mathematical structure and hence can form the theoretical basis for use in a wide range of applications. Second the models, when applied properly, work very well in practice for several important applications.\"

\"A problem of fundamental interest is characterizing such real-world signals in terms of signal models.\"

\"There are several reasons why one is interested in applying signal models. First of all, a signal model can provide the basis for a theoretical description of a signal processing system which can be used to process the signal so as to provide a desired output.\"

\"The underlying assumption of the statistical model is that the signal can be well characterized as a parametric random process, and that the parameters of the stochastic process can be determined (estimated) in a precise, well-defined manner.\"

## Problem Definition & Mathematical Framework

\"Consider a system which may be described at any time as being in one of a set of N distinct states, S₁, S₂, ..., Sₙ.\"

\"At regularly spaced discrete times, the system undergoes a change of state (possibly back to the same state) according to a set of probabilities associated with the state.\"

\"For the special case of a discrete, first order, Markov chain, this probabilistic description is truncated to just the current and the predecessor state.\"

\"We only consider those processes in which the right-hand side of (1) is independent of time, thereby leading to the set of state transition probabilities a_ij.\"

\"The above stochastic process could be called an observable Markov model since the output of the process is the set of states at each instant of time, where each state corresponds to a physical (observable) event.\"

## Hidden Markov Model Concept

\"In this section we extend the concept of Markov models to include the case where the observation is a probabilistic function of the state.\"

\"The resulting model (which is called a hidden Markov model) is a doubly embedded stochastic process with an underlying stochastic process that is not observable (it is hidden), but can only be observed through another set of stochastic processes that produce the sequence of observations.\"

\"This model is too restrictive to be applicable to many problems of interest.\"

## Methodology Description

\"We will first review the theory of Markov chains and then extend the ideas to the class of hidden Markov models using several simple examples.\"

\"We will then focus our attention on the three fundamental problems for HMM design, namely: the evaluation of the probability (or likelihood) of a sequence of observations given a specific HMM; the determination of a best sequence of model states; and the adjustment of model parameters so as to best account for the observed signal.\"

\"We will show that once these three fundamental problems are solved, we can apply HMMs to selected problems in speech recognition.\"

## Results & Discussion

\"We demonstrate empirically that our method accurately estimates variable importance and yields confidence intervals with asymptotically correct coverage.\"

\"The idea of characterizing the theoretical aspects of hidden Markov modeling in terms of solving three fundamental problems is due to Jack Ferguson of IDA (Institute for Defense Analysis) who introduced it in lectures and writing.\"

\"Neither the theory of hidden Markov models nor its applications to speech recognition is new.\"

\"Careful consideration of trade-offs between model complexity and predictive gains is crucial in algorithm selection.\"

## Reference Style

\"The basic theory was published in a series of classic papers by Baum and his colleagues in the late 1960s and early 1970s.\"

\"This tutorial is intended to provide an overview of the basic theory of HMMs (as originated by Baum and his colleagues), provide practical details on methods of implementation of the theory, and describe a couple of selected applications of the theory to distinct problems.\"
