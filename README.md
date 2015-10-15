# Odyssey
Odyssey is a public, GPU-based General Relativistic Radiative Transfer (GRRT) code for computing image and/or spectrum in Kerr spacetime. Implemented in CUDA C/C++, Odyssey is based on the ray-tracing algorithm presented in [Fuerst & Wu (2004)](http://adsabs.harvard.edu/abs/2004A%26A...424..733F), and radiative transfer formulation described in [Younsi et al. (2012)](http://adsabs.harvard.edu/abs/2012A%26A...545A..13Y).

For flexibility, namespace structure in C++  is used for different tasks. Two defalut tasks are presented in the sourse code. Including:
 <ol>
 <li>The redshift of the Keplerian disk</li>
 <li>The image of Keplerian rotating shell at 340GHz</li>
 </ol>
 The computed result is shown in the [wiki](https://github.com/hungyipu/Odyssey/wiki) . 
 
*This text will be italic*
**This text will be bold**
Here's an idea: why don't we take `SuperiorProject` and turn it into `**Reasonable**Project`
```
x = 0
x = 2 + 2
what is x
```
```ruby
require 'redcarpet'
markdown = Redcarpet.new("Hello World!")
puts markdown.to_html
```

| Left-Aligned  | Center Aligned  | Right Aligned |
| :------------ |:---------------:| -----:|
| col 3 is      | some wordy text | $1600 |
| col 2 is      | centered        |   $12 |
| zebra stripes | are neat        |    $1 |
## Summary of Source Codes
Odyssey source code prvided in the src folder includes the following files:
<ol>
 <li>main.cpp</li>
 assign parameters (black hole spin, inclinaiton angle, image size, observed frequency, CUDA configuration...), call Task, then save CUDA computed result to output file
 
 <li>task1.h</li>
 declare namespace for Task1
 
 
 <li>task1.cpp</li>
 define functions for setting up CUDA computation for Task1, such as allocate memory, copy memory between Host and Deevice, run CUDA computation, free memory

 
 <li>task2.h</li>
 declare namespace for Task2

 
 <li>task2.cpp</li>
  define functions for setting up CUDA computation for Task2, such as allocate memory, copy memory between Host and Deevice, run CUDA computation, free memory

 
 <li>Odyssey.cu</li>
 describe jobs of specific Task. Computation result will retun to main.cpp.
 
 <li>Odyssey_def.h</li>
 define constants (such as black hole mass, distance to the black hole),   
 <br />and variables which will be saved in the GPU global memory during computation
 
 
 <li>Odyssey_def_fun.h</li>
 define functions needed for
 <ul>
 <li>ray-tracing</li>
 such as initial condition, diffrential equaitons for geodesics, adaptive size Runge-Kutta method 
 <li>radiative transfer</li>
 such as table of Bessel function of the second kind (for computation of thermal synchoron emission), unit conversion to Jansky or Liminosity (erg/sec)
 </ul>
</ol>
Odyssey is a fast, accurate, and flexible code. Users can simply modifying the existing Tasks in Odyssey.cu by assigning different return value to main.cpp.

<br />To add a new Task (e.g., task 3), users can simply take Task1 and Task2 as example then
<ol>
 <li>add file: task3.h</li>
 <li>add file: task3.cpp</li>
 <li>add subroutine: task3() in main.cpp
 <li>describe job content in Odyssey.cu
</ol>



## Code Structure
The pseudo code of Odyssey is provided below.

main() `in main.cpp` <br />
task1() `in main.cpp` <br />
---assign parameters <br />
<br />
---set CUDA configuration `by setDims() defined`<br />
    |
    | --allocate memory on device for input and output *by PRE() defined*
    |
    | --perform the loop for performing GRRT

    | --copy memory form device to host and free CUDA memory *by AFTER() defined *
    |
    | --save result

## Reference
"Odyssey: A PUBLIC GPU-BASED CODE FOR GENERAL-RELATIVISTIC RADIATIVE TRANSFER IN KERR
SPACE-TIME"
