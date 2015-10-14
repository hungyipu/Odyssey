# Odyssey
Odyssey is a public, GPU-based GRRT (General Relativistic Radiative Transfer) code for computing image and/or spectrum in Kerr spacetime. Implemented in CUDA C/C++, Odyssey is based on the ray-tracing algorithm presented in [Fuerst & Wu (2004)](http://adsabs.harvard.edu/abs/2004A%26A...424..733F), and radiative transfer formulation described in [Younsi et al. (2012)](http://adsabs.harvard.edu/abs/2012A%26A...545A..13Y).

For flexibility, namespace structure in C++  is used for different tasks. Two defalut tasks are presented in the sourse code. Including:
<ol>
 <li>The red-shift of the Keplerian disk</li>
 <li>The image of Keplerian rotating shell at 340GHz</li>
 </ol>
 The computed result is shown in the [Wiki](https://github.com/hungyipu/Odyssey/wiki). 
 


## Summary of source codes
Odyssey source code provide in the github includes the following files:
<ul>
 <li>main.cpp</li>:dfasdfd
 <li>task1.cpp</li>
 <li>task1.h</li>
 <li>task2.cpp</li>
 <li>task2.h</li>
 <li>Odyssey.cu</li>
 <li>Odyssey_def.h</li>
 <li>Odyssey_def_fun.h</li>
</ul>




## Code Structure
The pseudo code of Odyssey is provided below.


## Reference
"Odyssey: A PUBLIC GPU-BASED CODE FOR GENERAL-RELATIVISTIC RADIATIVE TRANSFER IN KERR
SPACE-TIME"
