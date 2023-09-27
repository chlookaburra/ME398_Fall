# A Primer for Understanding Lagrangian Coherent Structures

Compiled by Tanner Harms

The study of Lagrangian Coherent Structures (LCS), while being rich in theory, conceptually beautiful, and pracitcally useful, can be daunting when approached as a lightly-informed initiate.  The purpose of these notes is not to probe the depths of any aspect of the LCS literature, but rather to provide new practitioners with the resources that they need to get started on their LCS journey.   While the rudimentary theory of the field will be considered, readers will primarily be prompted to peruse the literature themselves; links to the salient resources will be provided as they become relevant. Once the reader has become more familiar with the material, they are encouraged to tinker with the Python-based tutorial which comes with these notes.

I hope that those who read this will one day come to find the study of LCS as intriguing as I have.  Happy learning!

## Introduction

### What are Lagrangian Coherent Structures?

When I need to explain LCS to people without experience in fluid mechanics, I typically break it down into parts.  After all, "Lagrangian Coherent Structures" can sound a little intimidating if there is no context to back it up.  First, I tell people that Lagrangian refers to the way that the flow is studied.  Instead of considering the fluid as a continuum that passes through a fixed domain (the Eulerian frame), we instead consider it according to the motion of individual particles (or fluid parcels).  I tell them that, instead of thinking of the fluid in the context of the laboratory, we try to think of it from within its own frame of reference.  Adding "coherent structures" onto this simply means to study the *collective motion*  of regions of the fluid over time.  That is the idea of coherence--particles that evolve *together.*  Therefore, a layman's summary of the study of Lagrangian coherent structures is the segmentation of flows into regions with similar behavior by the examination individual and collective particle motion.

Conceptually this may seem relatively straightforward, but in fact, the definition of a coherent structure in the field of fluid mechanics remains contended.  Perhaps the most direct approach would be to define regions of large vorticity as coherent structures.  After all, it is intuitive that fluid elements near to the center of a vortex exhibit similar behavior to nearby elements.  This, however, has some difficulties (indeed, the definition of a vortex is even contended by experts!):  First, if we are considering the strength of the vorticity, how should we define the topology of the vortex?  Does the core correspond to the peak of vorticity?  How far from the vortex core do we draw the boundaries?  In fact, there are a variety of metrics that aim to address these questions.  Some prominent ones include the $q$-criterion (*****cite*****), the $\lambda_2$-criterion (***cite***), the Okubo-Weiss parameter (*****cite*****), and the $\Gamma_1$- and $\Gamma_2$-criterions (***cite***).  All of these metrics are defined using the velocity gradients of the flow (the velocity gradients are what vorticity is derived from, so they are closely related to the vorticity field of a flow) at a single moment--a snapshot--in time.

However in complex flows, vortices are formed, persist for some time, and then dissipate.  The snapshot (Eulerian) methods discussed above assume that the moment of observation is the moment when the structure is defined.  Is this an accurate assumption, though?  When during the lifespan of a vortex should a coherent structure really be defined?  To make matters even more complicated, the methods discussed above (and, of course, velocity and vorticity themselves) are sensitive to the relationship between the flow and the thing observing the flow.  That is to say, if my camera was rotating relative to the fluid in a vortex, the strength of the vorticity observed for the vortex would differ from if it were observed with a stationary camera.  In technical language, a metric which considers a duration longer than an infinitesimal snapshot (though not infinite time!) is said to have the property of *finite-time*, and one that yields the same results regardless of observer translation or rotation is said to be *objective*.  Finite-time and objectivity are hallmarks of the theory of LCS, and are used heavily throughout the literature.  Therefore, it is important to spend some time reflecting on them and internalizing what they represent.

As we progress in our study of LCS, we will see how many brilliant researchers have developed tools that respect objectivity and finite-time considerations to understand complex flows of all types.  However, before we dive any deeper, it is worth taking a brief look at the prominent literature listed below.  At the very least, take a look at the images!  Without a visual connection to the theory, developing the appropriate intuition will be quite difficult.

#### Introductory LCS Literature

* Haller Review Article
* Allshouse and Peacock Review Article
* Hadjighasem et al. Practical Review Article
* Haller and Yuan 2000 - First LCS paper
* Shadden book chapter

### Approaches for Approximating LCS

#### Advantages and Disadvantages

## Geometric LCS

### The Setting of Geometric LCS

### Geometric LCS Literature

## Probabilistic LCS

### The Setting of Probabilistic LCS

### Probabilistic LCS Literature

## Sparse Methods

## Recent Developments

### Objective Eulerian Coherent Structures

#### OECS Literature

### Quasi-Objective Single-Particle Metrics

#### Quasi-Objective Literature

### Lagrangian Gradient Regression

#### LGR Literature

## Outlook
