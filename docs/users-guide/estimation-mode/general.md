
# Estimation Mode

ActivitySim includes the ability to re-estimate submodels using choice model estimation
tools. It is possible to output the data needed for estimation and then use more or less
any parameter estimation tool to find the best-fitting parameters for each model, but
ActivitySim has a built-in integration with the [`larch`](https://larch.driftless.xyz)
package, which is an open source Python package for estimating discrete choice models.

## Estimation Workflow, Summarized

The general workflow for estimating models is shown in the following figures and
explained in more detail below.

![estimation workflow](https://activitysim.github.io/activitysim/develop/_images/estimation_tools.jpg)

First, the user converts their household travel survey into ActivitySim-format
households, persons, tours, joint tour participants, and trip tables.  The
households and persons tables must have the same fields as the synthetic population
input tables since the surveyed households and persons will be run through the same
set of submodels as the simulated households and persons.

The ActivitySim estimation example [``scripts\infer.py``](https://github.com/ActivitySim/activitysim/blob/main/activitysim/examples/example_estimation/scripts/infer.py)
module reads the ActivitySim-format household travel survey files and checks for
inconsistencies in the input tables versus the model design, and calculates
additional fields such as the household joint tour frequency based on the trips
and joint tour participants table.  Survey households and persons observed choices
much match the model design (i.e. a person cannot have more work tours than the model
allows).

ActivitySim is then run in estimation mode to read the ActivitySim-format
travel survey files, and apply the ActivitySim submodels to write estimation data bundles
(EDBs) that contains the model utility specifications, coefficients, chooser data,
and alternatives data for each submodel.

The relevant EDBs are read and transformed into the format required by the model
estimation tool (i.e. larch) and then the coefficients are re-estimated. The
``activitysim.estimation.larch`` library is included for integration with larch
and there is a Jupyter Notebook estimation example for most core submodels.
Certain kinds of changes to the model specification are allowed during the estimation
process, as long as the required data fields are present in the EDB.  For example,
the user can add new expressions that transform existing data, such as converting
a continuous variable into a categorical variable, a polynomial transform, or a
piecewise linear form.  More intensive changes to the model specification, such as
adding data that is not in the EDB, or adding new alternatives, are generally not
possible without re-running the estimation mode to write a new EDB.

Based on the results of the estimation, the user can then update the model
specification and coefficients file(s) for the estimated submodel.

## Estimating ActivitySim Models with Larch

ActivitySim component models are mostly built as discrete choice models.  The
parameters for these models typically need to be estimated based on observed
survey data.  The estimation process is facilitated by the Larch package, which
is a Python package for estimating discrete choice models.  Larch is a
general-purpose package that can be used to estimate a wide variety of discrete
choice models, including the multinomial logit and nested logit models that
are commonly used in ActivitySim.

### Maximum Likelihood Estimation

The approach used to estimate the parameters of a discrete choice model is
maximum likelihood estimation (MLE).  The goal of MLE is to find the set of
parameters that maximize the likelihood of observing the choice data that we
have collected.

Finding the maximum likelihood estimates of the parameters is a non-linear
optimization problem.  To solve this problem, Larch primarily relies on the
widely-used `scipy.optimize` package, which provides a number of
[optimization algorithms](https://docs.scipy.org/doc/scipy/reference/optimize.html#local-multivariate-optimization)
that can be used to find the maximum likelihood estimates. Different algorithms
have different strengths and weaknesses, and the choice of algorithm can have a
significant impact on the speed and accuracy of the estimation process.  By default,
when no constraints or bounds are present, Larch uses an implementation of the
[BHHH algorithm](https://en.wikipedia.org/wiki/Berndt–Hall–Hall–Hausman_algorithm),
which is not included in scipy but is usually efficient for simple
choice models.  When constraints or bounds are present, by default Larch uses the
Larch uses the `scipy.optimize.minimize` function with the `SLSQP` algorithm when
constraints or bounds are present.  The `larch.Model.estimate` method allows the
user to specify the optimization algorithm to use via the `method` argument, which
can be set to 'BHHH', 'SLSQP', or any other algorithm supported by `scipy.optimize.minimize`.
If you are estimating a model and find the optimization is not converging as
fast as expected (or at all), you may want to try a different optimization algorithm.

### Expressing Alternative Availability

In ActivitySim, the unavailability of alternatives is typically expressed in the
utility function given in the model specification, by including a indicator variable
for unavailable alternatives, which is then attached to a large negative coefficient.
This creates a large negative utility for the unavailable alternative, which will
render it effectively unavailable in the choice model.  If *all* the alternatives
are made unavailable in this manner, this can result in a condition where no
alternative can be chosen, and ActivitySim will raise an error.

Larch, on the other hand, does typically use this approach to express alternative
availability.  Instead, Larch defines the availability of alternatives as a separate
array of values, which is not included in the utility function.  This is typically
more robust in estimation, as the computational engine can automatically shift
the utility values to avoid numerical underflow or overflow issues that can arise
when some choices are very unlikely but not strictly unavailable.

## Example Notebooks

ActivitySim includes a collection of Jupyter notebooks with
interactive re-estimation examples for many core submodels, which can be found in the
GitHub repository under the [`activitysim/examples/example_estimation/notebooks`](https://github.com/ActivitySim/activitysim/tree/main/activitysim/examples/example_estimation/notebooks)
directory.  Most of these notebooks demonstrate the process of re-estimating model
parameters, without changing the model specification, i.e. finding updated values
for coefficients without changing the mathematical form of a model's utility
function.

A selection of these notebooks have also been updated to demonstrate
the process of estimating model parameters and also *changing the model specification*.
These notebooks generally include instrucations and a demonstration of how to
modify the model specification, and then re-estimate the model parameters, as
well as how to compare the results of the original and modified models side-by-side,
which can be useful for understanding the impact of the changes made, and conducting
statistical tests to determine if the changes made are statistically significant.

The following notebooks include examples of modifying the model specification:

### `03_work_location.ipynb`

This notebook includes a demonstration of modification to the SPEC file for a
destination choice model, using the "interact-sample-simulate" type model.

### `04_auto_ownership.ipynb`

This notebook includes a demonstration of modification to the SPEC file for the
auto ownership model. It shows an example of an edit in the utility function
for a "simple simulate" type model.

### `06_cdap.ipynb`

This notebook includes a demonstration of modification to the SPEC file for the
CDAP model. This model has a complex structure that is unique among the
ActivitySim component models.
