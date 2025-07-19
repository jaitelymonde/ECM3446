/*******************************************************************************
2D advection example program which advects a Gaussian u(x,y) at a fixed velocity



Outputs: initial.dat - inital values of u(x,y) 
         final.dat   - final values of u(x,y)

         The output files have three columns: x, y, u

         Compile with: gcc -o advection2D -std=c99 advection2D.c -lm

Notes: The time step is calculated using the CFL condition

********************************************************************************/

/*********************************************************************
                     Include header files 
**********************************************************************/

#include <stdio.h>
#include <math.h>
#include <omp.h>

/*********************************************************************
                      Main function
**********************************************************************/

int main(){

  /* Grid properties */
  const int NX=1000;    // Number of x points
  const int NY=1000;    // Number of y points
  //change comp domain from 0-x-1; 0-y-1 to 0-x-30; 0-y-30
  const float xmin=0.0; // Minimum x value
  const float xmax=30.0; // Maximum x value
  const float ymin=0.0; // Minimum y value
  const float ymax=30.0; // Maximum y value
  
  /* Parameters for the Gaussian initial conditions */
  //change centre of gaussian to x0=3.0; y0=15.0
  const float x0=3.0;                    // Centre(x)
  const float y0=15.0;                    // Centre(y)
  //change gaussian width to 1.0 and 5.0
  const float sigmax=1.0;               // Width(x)
  const float sigmay=5.0;               // Width(y)
  const float sigmax2 = sigmax * sigmax; // Width(x) squared
  const float sigmay2 = sigmay * sigmay; // Width(y) squared

  /* Boundary conditions */
  const float bval_left=0.0;    // Left boudnary value
  const float bval_right=0.0;   // Right boundary value
  const float bval_lower=0.0;   // Lower boundary
  const float bval_upper=0.0;   // Upper bounary

  /* Time stepping parameters */
  const float CFL=0.9;   // CFL number 
  //change time steps to 800
  const int nsteps=800; // Number of time steps

  /* 2.3 - task 3 variables for adding vertical shear*/
  const float fvel=0.2; //u* - friction velocity (m/s)
  const float rlen=1.0; //z0 - roughness length (m)
  const float vkc=0.41; //Von K'arm'an's constant

  /* Velocity */
  //change horizontal velocity to 1.0 and vertical to 0
  const float velx=1.0; // Velocity in x direction
  const float vely=0.0; // Velocity in y direction
  
  /* Arrays to store variables. These have NX+2 elements
     to allow boundary values to be stored at both ends */
  float x[NX+2];          // x-axis values
  float y[NX+2];          // y-axis values
  float u[NX+2][NY+2];    // Array of u values
  float dudt[NX+2][NY+2]; // Rate of change of u

  float x2;   // x squared (used to calculate iniital conditions)
  float y2;   // y squared (used to calculate iniital conditions)
  
  /* Calculate distance between points */
  float dx = (xmax-xmin) / ( (float) NX);
  float dy = (ymax-ymin) / ( (float) NY);
  
  /* Calculate time step using the CFL condition */
  /* The fabs function gives the absolute value in case the velocity is -ve */
  float dt = CFL / ( (fabs(velx) / dx) + (fabs(vely) / dy) );
  
  /*** Report information about the calculation ***/
  printf("before task 3:");
  printf("\nGrid spacing dx     = %g\n", dx);
  printf("Grid spacing dy     = %g\n", dy);
  printf("CFL number          = %g\n", CFL);
  printf("Time step           = %g\n", dt);
  printf("No. of time steps   = %d\n", nsteps);
  printf("End time            = %g\n", dt*(float) nsteps);
  printf("Distance advected x = %g\n", velx*dt*(float) nsteps);
  printf("Distance advected y = %g\n", vely*dt*(float) nsteps);

  /*** Place x points in the middle of the cell ***/
  /* LOOP 1 */
  #pragma omp parallel for shared(x,dx)
  for (int i=0; i<NX+2; i++){
    x[i] = ( (float) i - 0.5) * dx;
  }

  /*** Place y points in the middle of the cell ***/
  /* LOOP 2 */
  #pragma omp parallel for shared(y,dy)
  for (int j=0; j<NY+2; j++){
    y[j] = ( (float) j - 0.5) * dy;
  }

  /*** Set up Gaussian initial conditions ***/
  /* LOOP 3 */
  #pragma omp parallel for collapse(2) private(x2,y2) shared(x,y,u)
  for (int i=0; i<NX+2; i++){
    for (int j=0; j<NY+2; j++){
      x2      = (x[i]-x0) * (x[i]-x0);
      y2      = (y[j]-y0) * (y[j]-y0);
      u[i][j] = exp( -1.0 * ( (x2/(2.0*sigmax2)) + (y2/(2.0*sigmay2)) ) );
    }
  }

  //2.3 - task 3
  //set new horizontal velocity as an array
  float h_vel[NY+2];
  //set max horizontal velocity variable
  float max_h_vel = velx;
  #pragma omp parallel for shared(h_vel,y) reduction(max:max_h_vel)
  for (int j=1; j<NY+1; j++){
    //if z > z0
    if (y[j] > rlen){
      //horizontal velocity by height ([j) = (friction vel/von karm const)ln(height/roughness length) 
      //vx(z) = (u*/vkc)ln(z/z0)
      h_vel[j] = (fvel/vkc) * log((y[j])/rlen);
    }else{ //if z <= z0 set velocity to 0
      h_vel[j] = 0.0;
    }
    //set max horizontal velocity
    max_h_vel = fmaxf(max_h_vel, h_vel[j]);
  }
  
  //recalculate dt with CFL
  dt = CFL / ((max_h_vel / dx) + (fabs(vely) / dy));

  printf("\nafter task 3:");
  printf("\nGrid spacing dx     = %g\n", dx);
  printf("Grid spacing dy     = %g\n", dy);
  printf("CFL number          = %g\n", CFL);
  printf("Time step           = %g\n", dt);
  printf("No. of time steps   = %d\n", nsteps);
  printf("End time            = %g\n", dt*(float) nsteps);
  printf("Distance advected x = %g\n", max_h_vel*dt*(float) nsteps);
  printf("Distance advected y = %g\n", vely*dt*(float) nsteps);


  /*** Write array of initial u values out to file ***/
  FILE *initialfile;
  initialfile = fopen("initial.dat", "w");
  /* LOOP 4 */
  /*not paralellised due to filewriting, can mess up order of output when not output to file sequentially*/
  for (int i=0; i<NX+2; i++){
    for (int j=0; j<NY+2; j++){
      fprintf(initialfile, "%g %g %g\n", x[i], y[j], u[i][j]);
    }
  }
  fclose(initialfile);
  
  /*** Update solution by looping over time steps ***/
  /* LOOP 5 */
  //not parallelised - sequential dependancy from m needing m-1 values, must be executed sequentially
  for (int m=0; m<nsteps; m++){

    /*** Apply boundary conditions at u[0][:] and u[NX+1][:] ***/
    /* LOOP 6 */
    #pragma omp parallel for shared(u)
    for (int j=0; j<NY+2; j++){
      u[0][j]    = bval_left;
      u[NX+1][j] = bval_right;
    }

    /*** Apply boundary conditions at u[:][0] and u[:][NY+1] ***/
    /* LOOP 7 */
    #pragma omp parallel for shared(u)
    for (int i=0; i<NX+2; i++){
      u[i][0]    = bval_lower;
      u[i][NY+1] = bval_upper;
    }
    
    /*** Calculate rate of change of u using leftward difference ***/
    /* Loop over points in the domain but not boundary values */
    /* LOOP 8 */
    #pragma omp parallel for collapse(2) shared(dudt,dx,dy,u,h_vel)
    for (int i=1; i<NX+1; i++){
      for (int j=1; j<NY+1; j++){

      //replace velx with h_vel[j]
	  dudt[i][j] = -h_vel[j] * (u[i][j] - u[i-1][j]) / dx
	            - vely * (u[i][j] - u[i][j-1]) / dy;
      }
    }
    
    /*** Update u from t to t+dt ***/
    /* Loop over points in the domain but not boundary values */
    /* LOOP 9 */
    #pragma omp parallel for collapse(2) shared(dudt,dt,u)
    for	(int i=1; i<NX+1; i++){
      for (int j=1; j<NY+1; j++){
	u[i][j] = u[i][j] + dudt[i][j] * dt;
      }
    }
    
  } // time loop
  
  /*** Write array of final u values out to file ***/
  FILE *finalfile;
  finalfile = fopen("final.dat", "w");
  /* LOOP 10 */
  /*not paralellised due to filewriting, can mess up order of output when not output to file sequentially*/
  for (int i=0; i<NX+2; i++){
    for (int j=0; j<NY+2; j++){
      fprintf(finalfile, "%g %g %g\n", x[i], y[j], u[i][j]);
    }
  }
  fclose(finalfile);

  /* 2.4 - task 4 */
  /*** Write array of averaged u values out to file ***/
  FILE *averagefile;
  averagefile = fopen("average.dat", "w");
  for (int i=1; i<NX+1; i++){
    //initialise average
    float u_avg = 0.0;
    //loop through u values
    for (int j=1; j<NY+1; j++){
      u_avg += u[i][j];
    }
    //divide by y points to get average
    //exclude boundary
    u_avg /= (float)(NY-2);
    //write to averagefile
    fprintf(averagefile, "%g %g\n", x[i], u_avg);
  }
  //close file
  fclose(averagefile);

  return 0;
}

/* End of file ******************************************************/