.. Geom-DeepONet documentation master file, created by
   sphinx-quickstart on Tue Dec 16 15:00:25 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

deeponet-for-mems documentation
===============================

.. Add your content using ``reStructuredText`` syntax. See the
.. `reStructuredText <https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html>`_
.. documentation for details.
.. _abstract:

Abstract
--------

This project tackles the simulation of coupled electrostatic and mechanical phenomena in MEMS (Micro-Electro-Mechanical Systems) devices, whose widespread adoption in contemporary technologies makes accurate yet efficient modeling essential.

Full-order numerical solvers quickly become computationally prohibitive for realistic geometries, and in such cases one of the possible solutions could be relying on reduced-order modeling.

Indeed, we choose to do so: we construct an idealized test case that preserves the key physical features, then we develop a reduced-order model based on recent Deep Learning–based ROM (DL-ROM) techniques, in particular DeepONets, exploiting an approximation of mechanical deformation grounded in Euler–Bernoulli beam theory in order to design it to interface with a finite-element solver for the mechanical response.

A brief literature review frames our methodology within current trends in MEMS modeling and data-driven model reduction.

**Tools:**

- ``gmsh`` has been used for geometry and meshing
- FEniCSx for high-fidelity data generation
- TensorFlow/Keras for the network architecture
- Sphinx for the documentation

**Keywords:** DeepONet, DL-ROM, MEMS, Euler-Bernoulli beam theory

**Authors:** Alessandro Pedone, Marta Pignatelli

Contents
--------

.. toctree::
   :maxdepth: 1
   
   README
   run_test_cases
   data
   surrogate
   multi-physics