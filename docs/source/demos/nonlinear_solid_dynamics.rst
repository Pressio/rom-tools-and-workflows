Solid dynamics model
====================

The examples include a lightweight two-dimensional solid-dynamics solver
written in pure Python with NumPy and SciPy.  The model is intended as a
transparent reference problem for projection-based ROMs, hyper-reduction,
surrogates, UQ, and inverse workflows.  It is not intended to replace a
production structural-mechanics code.

The core Neo-Hookean solver, the common material-model facade, and the romtools
QoI wrapper live in ``examples/models``.  The user-facing examples live in
``examples/nonlinear_solid_dynamics``.

Governing equations
-------------------

After finite-element discretization the displacement satisfies

.. math::

   M \ddot{u} + C \dot{u} + f_{\mathrm{int}}(u;\mu)
   = f_{\mathrm{ext}}(t;\mu).

Both displacement and velocity are retained as state variables.  The mechanics
operators are exposed independently of the time integrator through methods such
as ``mass_matrix``, ``internal_force``, ``tangent_stiffness``, and
``acceleration``.  Element-level internal-force evaluation is also available
for future hyper-reduction examples.

Spatial discretization and materials
------------------------------------

The implementation uses a structured rectangular mesh of bilinear Q4 elements,
two-by-two Gauss quadrature, and plane-strain kinematics.  Two constitutive
models are available through ``examples/models/solid_dynamics.py``.

Neo-Hookean
^^^^^^^^^^^

The nonlinear model uses a total-Lagrangian formulation and the compressible
Neo-Hookean energy

.. math::

   W(C,J) = \frac{\lambda}{4}(J^2-1)
   - \left(\frac{\lambda}{2}+\mu\right)\log J
   + \frac{\mu}{2}(I_C-3),

where ``lambda`` and ``mu`` are the Lamé constants.  The corresponding first
Piola stress is

.. math::

   P = \mu F
   + \left[\frac{\lambda}{2}(J^2-1)-\mu\right]F^{-T}.

The consistent material tangent is assembled analytically.

Linear elasticity
^^^^^^^^^^^^^^^^^

A small-strain isotropic linear-elastic option is also available.  It uses

.. math::

   \varepsilon = \frac{1}{2}\left(\nabla u + \nabla u^T\right),

.. math::

   \sigma = \lambda\,\mathrm{tr}(\varepsilon)I + 2\mu\varepsilon.

The resulting internal force is linear in displacement and the tangent
stiffness is constant.  The linear material reuses the same mesh, mass matrix,
boundary-condition handling, and explicit/implicit time integrators as the
Neo-Hookean material.  The beam and doubly clamped examples accept
``material_model="neo_hookean"`` or ``material_model="linear"``.

Both consistent and row-sum lumped mass matrices are supported.

Time integration
----------------

``solve_explicit`` uses velocity Verlet.  ``solve_implicit`` uses Newmark with
the average-acceleration defaults

.. math::

   \gamma=\frac{1}{2}, \qquad \beta=\frac{1}{4}.

The implicit formulation deliberately uses **velocity at the new time step as
the Newton unknown**.  For a trial ``v_{n+1}``,

.. math::

   a_{n+1} = \frac{v_{n+1}-v_n}{\gamma\Delta t}
   - \frac{1-\gamma}{\gamma}a_n,

and ``u_{n+1}`` follows from the Newmark displacement relation.  The residual is

.. math::

   R(v_{n+1}) = M a_{n+1} + C v_{n+1}
   + f_{\mathrm{int}}(u_{n+1}) - f_{\mathrm{ext}}(t_{n+1}),

with Jacobian

.. math::

   \frac{\partial R}{\partial v}
   = \frac{1}{\gamma\Delta t}M + C
   + \frac{\beta\Delta t}{\gamma}K_{\mathrm{tan}}.

Newton iterations use a simple backtracking line search.  For the linear
material the same formulation is retained, although the residual is affine and
one exact Newton correction is sufficient up to numerical roundoff.

Examples
--------

Cantilever beam
^^^^^^^^^^^^^^^

``examples/nonlinear_solid_dynamics/beam.py`` clamps the left end of a
rectangular beam and applies a smooth transient transverse resultant on the
right end.  The example plots the vertical tip-displacement history and stores
full displacement/velocity trajectories in memory for direct use as ROM
snapshots.  Its ``run`` function accepts either constitutive model.

Run the default Neo-Hookean case with

.. code-block:: bash

   python examples/nonlinear_solid_dynamics/beam.py

Doubly clamped Gaussian perturbation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``examples/nonlinear_solid_dynamics/clamped_gaussian.py`` fixes both ends and
starts from a smooth transverse Gaussian displacement centered in the domain,

.. math::

   u_y(X,0) = A\exp\left[-\frac{(X-X_c)^2}{2\sigma^2}\right].

The amplitude, width, and center are explicit parameters of the initialization.
The same example can use either the Neo-Hookean or linear-elastic material.

The **linear transverse problem is not the scalar two-way wave equation**.  A
transverse disturbance of a clamped beam excites flexural/elastic modes; in a
slender-beam approximation the corresponding Euler--Bernoulli equation contains
a fourth spatial derivative and is dispersive.  In the full 2-D linear solid,
the governing equations are the vector elastodynamic equations and support
longitudinal and shear waves.

Longitudinal two-way wave check
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``examples/nonlinear_solid_dynamics/linear_wave.py`` provides a separate
configuration that does reduce to the scalar two-way wave equation.  It uses the
linear material, constrains all transverse DOFs, fixes the axial displacement at
both ends, and initializes an axial Gaussian perturbation that is uniform
through the height.  The continuum reduction is

.. math::

   u_{tt} = c_p^2 u_{xx},

with the plane-strain longitudinal wave speed

.. math::

   c_p = \sqrt{\frac{\lambda+2\mu}{\rho}}.

An initially stationary Gaussian therefore splits into equal left- and
right-traveling pulses before boundary reflections occur.  This provides a
simple linear-dynamics sanity check independent of the nonlinear benchmark.

Run it with

.. code-block:: bash

   python examples/nonlinear_solid_dynamics/linear_wave.py

Published nonlinear benchmark check
-----------------------------------

``examples/nonlinear_solid_dynamics/benchmark_cantilever.py`` reproduces the
large-deformation cantilever benchmark of
`Stickle et al. (2022) <https://doi.org/10.1007/s00466-021-02107-0>`_.  The
published problem is a 4 m by 1 m plane-strain Neo-Hookean cantilever with

.. math::

   E=1\ \mathrm{MPa},\qquad \nu=0.3,\qquad
   \rho_0=1050\ \mathrm{kg/m^3},\qquad g=10\ \mathrm{m/s^2}.

The left end is clamped and gravity is applied instantaneously at ``t=0``.  The
reference calculation uses Newmark with ``beta=0.25``, ``gamma=0.5`` and
``dt=1e-3 s``.  The paper uses an 8 by 2 Q8 mesh for the plotted benchmark;
the romtools reference implementation intentionally uses Q4 elements.

For the 8 by 2 Q4 mesh, the present implementation gives a first lower-right
vertical-displacement minimum of approximately ``-3.17 m`` at ``t=1.00 s``.
The published response reaches approximately ``-3.3 m`` near ``t=1.0--1.1 s``.
The agreement is sufficient for the intended lightweight implementation check,
particularly given the difference in element order.

A short CI regression uses ``dt=1e-2 s`` through the first displacement minimum.
The full example retains the published ``dt=1e-3 s`` and 3 s time horizon.

Verification scope
------------------

The initial solver keeps verification deliberately focused:

* zero internal force is checked in the undeformed configuration;
* the Neo-Hookean analytical tangent is compared with a finite-difference
  directional derivative of the internal force;
* the linear material is checked for force linearity and a constant tangent;
* the longitudinal linear-wave configuration is checked for the expected
  left/right symmetry;
* a small velocity-primary Newmark solve checks constrained DOFs and nonlinear
  convergence; and
* the first minimum of the Stickle cantilever response is used as an external
  nonlinear regression check.

These checks establish confidence in the example implementation without turning
romtools into a general solid-mechanics verification suite.

QoI wrapper
-----------

``NonlinearSolidBeamQoiModel`` in
``examples/models/nonlinear_solid_dynamics_model.py`` exposes a parameterized
cantilever through the standard romtools model protocol.  The sample parameters
are ``young_modulus``, ``load_amplitude``, and ``pulse_duration``.  The QoI is
the vertical tip-displacement history; full state snapshots are saved alongside
the QoI for later ROM construction.  The wrapper constructor also accepts
``material_model="neo_hookean"`` or ``material_model="linear"``.

Implementation
--------------

Neo-Hookean core solver:

.. literalinclude:: ../../../examples/models/nonlinear_solid_dynamics.py
   :language: python
   :linenos:

Material-model facade and linear elasticity:

.. literalinclude:: ../../../examples/models/solid_dynamics.py
   :language: python
   :linenos:

romtools QoI wrapper:

.. literalinclude:: ../../../examples/models/nonlinear_solid_dynamics_model.py
   :language: python
   :linenos:
