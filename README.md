# Odyssey
Odyssey is a public, GPU-based General Relativistic Radiative Transfer (GRRT) code for computing image and/or spectrum in Kerr spacetime. Implemented in CUDA C/C++, Odyssey is based on the ray-tracing algorithm presented in [Fuerst & Wu (2004)](http://adsabs.harvard.edu/abs/2004A%26A...424..733F), and radiative transfer formulation described in [Younsi et al. (2012)](http://adsabs.harvard.edu/abs/2012A%26A...545A..13Y).

For flexibility, namespace structure in C++  is used for different tasks. Two defalut tasks are presented in the sourse code. Including:

 *The redshift of the Keplerian disk</li>
 *The image of Keplerian rotating shell at 340GHz</li>
 
 The computed result is shown in the [wiki](https://github.com/hungyipu/Odyssey/wiki) . 
 
Inline-style: 
![alt text](https://github.com/adam-p/markdown-here/raw/master/src/common/images/icon48.png "Logo Title Text 1")

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
Odyssey source code prvided in the src folder includes the following files:<br />
1. **main.cpp**
assign parameters (black hole spin, inclinaiton angle, image size, observed frequency, CUDA configuration...), call Task, then save CUDA computed result to output file<br />
2. **main.cpp**
<ol>
 1. **main.cpp**<br />
 assign parameters (black hole spin, inclinaiton angle, image size, observed frequency, CUDA configuration...), call Task, then save CUDA computed result to output file
 
 <li>**task1.h**</li>
 declare namespace for Task1
 
 
 <li>**task1.cpp**</li>
 define functions for setting up CUDA computation for Task1, such as allocate memory, copy memory between Host and Deevice, run CUDA computation, free memory

 
 <li>**task2.h**</li>
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


## Code Structure
The flow chart for the code structure of Odyssey is provided below.

In **main.cpp**, `task1()` is called by `main()`, then go through<br />
<br />
`task1()`:<br />
|---assign parameters <br />
|<br />
|---set CUDA configuration `setDims()`<br />
|<br />
|---allocate memory on device for input and output `PRE()`<br />
|<br />
|---perform the *for loop* for performing GRRT `GPUcompute()`<br />
|<br />
|---copy memory form device to host and free CUDA memory `by AFTER()`<br />
|<br />
|---save result<br />
## Code Structure: more details
By calling `GPUcompute()`, the parallel computation job detial is finally assigned to  `__global__ GPU_task1work()` in **Odyssey.cu**, thern go through <br />
<br />
`__global__ GPU_task1work()`:<br />
|---assign parameters <br />
|<br />
|---set CUDA configuration `setDims()`<br />
|<br />
|---allocate memory on device for input and output `PRE()`<br />
|<br />
|---perform the *for loop* for performing GRRT `GPUcompute()`<br />
|<br />
|---copy memory form device to host and free CUDA memory `by AFTER()`<br />
|<br />
|---save result<br />
<br />
Odyssey is a fast, accurate, and flexible code. Users can simply modifying the existing Tasks in **Odyssey.cu** by assigning different return value to **main.cpp**.
<br />To add a new Task (e.g., task 3), users can simply take Task1 and Task2 as examples, then
 1. add file: task3.h</li>
 2. add file: task3.cpp</li>
 3. add subroutine: task3() in **main.cpp**
 4. describe job details in **Odyssey.cu**
 
## Reference
"Odyssey: A PUBLIC GPU-BASED CODE FOR GENERAL-RELATIVISTIC RADIATIVE TRANSFER IN KERR
SPACE-TIME"
