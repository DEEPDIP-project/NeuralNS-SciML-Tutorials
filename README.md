# Neural closures using SciML

In this repository, you can see two sets of tutorials about fluid dynamics with NeuralODE and the SciML Julia Library. They are divided according to the problem that they are targeting, in specific:
* `NS_*` tutorials show how to solve the Incompressible Navier-Stokes equation in the *spectral space*. They are based on Syver's implementation.
* `Adv_*` tutorials focus on advection problems (including Burgers). They are based on Toby's code.
  
We will look at the two groups separately, but later we plan to merge them in a single framework.

## Navier-Stokes tutorials
* In `NS_SciML_vs_direct.jl` you can see the comparison between different timestep solvers. In the SciML community it is suggested to use `Tsit5`, but we can use a Runge-Kutta approach to compare with our (Syver) direct implementation:

![Alt text](plots/NS_SciML_vs_direct.gif)

* In `NS_data_generation.jl` you can generate DNS and LES data, to train multiple Neural closures.
  
* In `NS_train_closure_model.jl` you can specify the type of closure that you would like to train. It is also very easy to implement new closures (such as [CNO](https://github.com/bogdanraonic3/ConvolutionalNeuralOperator)) following the template of the other closures.

* In `NS_test_closure.jl` you can visualize the performance of a specific trained closure:
  
![Alt text](plots/FNO__2-5-5-5-2__8-8-8-8__gelu-gelu-gelu-identity_lossDtO-nu20_DNS_128_LES_64_nu_0.0005_1234.gif)

* In `NS_compare_closures.jl` you can compare all the different closures that you have trained for a specific problem. This allow you to easily visualize which closure approach performs better:
  
![Alt text](plots/DNS_128_LES_64_nu_0.0005_1234_error.png)
*(Disclaimer: the models in the figure have been '''trained''' for ~10m of single cpu time)*


## Advection tutorials

* In `Adv_SciML_vs_direct.jl` you can see the comparison between different timestep solvers. In the SciML community it is suggested to use `Tsit5`, but we can use a Runge-Kutta approach to compare with our (Toby) direct implementation:

![Alt text](plots/Adv_SciML_vs_direct.gif)

* ...Work in progress...
