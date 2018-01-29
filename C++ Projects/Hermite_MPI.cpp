//EMPLID: 200344273
//Q8, Dr. Huang, Summer/August 2017
//Cubic Hermite Integration
//9-3-2016
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <math.h>
#include <stdio.h>

#include <mpi.h>


using namespace std;

//Definition of function f(x)
double f(double x)
{
	double fOutput = x + (5 * x * x) - (0.5 * x * x * x) ;
	return fOutput;    
}




int main(int argc, char *argv[]) 
{
	//Defining the parameters
	const double PI = 3.141592653589793238462643383279502884;
  	double a;
	double aa;
  	double b;
  
	//mpi variables
	int id;
	int p;
	double size = 10;
	double f_swap_plus;
	double f_swap_minus;
	double local_sum;

	double mx;
	double mx_plus;
	double ma;
	double mb;
	
	double wtime = 0.0;
	double Area_sum;

	int tally;
       	MPI_Status status;

//EMPLID: 200344273

	a = 0;
	b = 10;

	int rank,length;
	char name[BUFSIZ];
	
  	
	//MPI region
 	MPI_Init (&argc, &argv);
  	MPI_Comm_size (MPI_COMM_WORLD, &p);
	MPI_Comm_rank (MPI_COMM_WORLD, &rank);	
	MPI_Get_processor_name(name,&length);

	//cout << rank << endl;

  	int offset = ( size / p);
  	a = (rank*offset);
  	b = (rank+1)*(offset);

	///////////// Sending out f(xi)		

	if (rank != 0)
	{
		f_swap_plus  = f(a);
		//Sending f(xi) to (fxi-1)
		MPI_Send(&f_swap_plus, offset, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD);
		//  <==============
	}

	if (rank != p-1) //if rank is less than p-1 (processors count 1 to n )
	{
	    	f_swap_minus = f(a);
	    	//Sending f(xi) to f(xi+1)
	    	MPI_Send(&f_swap_minus, offset, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD);
	    	//  =============>
	}

	//////////computing m(xi)

	if (rank == p-1) // if processor is the last in line
	{
	    	MPI_Recv(&f_swap_minus, offset, MPI_DOUBLE, rank - 1, 0 , MPI_COMM_WORLD, &status); //recieve f_swap_minus

	    	//finding x10
		double x_eight = ( (rank-1) *  offset); // x8
		double x_nine  = a;                     // x9
	    	double x_ten   = ( (rank+1)  * offset); // x10
				
//EMPLID: 200344273

	    	//find m(x9), you need x8 and x10
	    	double mx_nine =  ( f(x_ten) - f_swap_minus ) / ( x_ten - x_eight ) ;
	   	
	    	//finding m(x10), you need x10 and x
		double mx_ten = ( f(x_ten) - f(x_nine) ) / ((x_ten) - (x_nine)); 		
		
	    	//f(a)  = f(x_nine)
	    	//f(b)  = f(x_ten)
	    	local_sum = (.25) * ( 2*f(x_nine) + mx_nine - 2*f(x_ten) + mx_ten) + (1/3) * (-3*f(x_nine) - 2*mx_nine + 3*f(x_nine)-mx_ten) + (.5)*(mx_nine) + f(x_nine);
		mx_plus = mx_nine;
		MPI_Send(&mx_plus, offset, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD);
	}



	if(rank < p-1) // all processors other than the last one
     	{

	    	MPI_Recv(&f_swap_plus, offset, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &status);
		
	    	//getting m_local_sum for first processor
		if (rank == 0)
		{	    	
			mx  = (f(a) - f_swap_plus) / (a - b);
		}
		
		//cout << "Rank " << rank << " Ding " << endl;

	    	// if it is any other processor besides the first, m_local_sum gets overwritten 
	    	if (rank != 0)
	      	{
			ma = (rank-1) * offset;
			mb = (rank+1) * offset;
			
			MPI_Recv(&f_swap_minus, offset, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &status);
			mx = ( f_swap_plus - f_swap_minus ) / ( mb - ma );

			mx_plus = mx;
			MPI_Send(&mx_plus, offset, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD);	
	      	}



//EMPLID: 200344273
         
	    	MPI_Recv(&mx_plus, offset, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &status);                       
		local_sum = (.25)*(2*f(a)+mx-2*f_swap_plus+mx_plus)+(1/3)*(-3*f(a)-2*mx + 3*f_swap_plus-mx_plus) + (.5)*(mx) + f(a);
	}

  	double global_sum;

	//Reducing
	MPI_Reduce(&local_sum, &global_sum, 1 , MPI_DOUBLE, MPI_SUM, 0 , MPI_COMM_WORLD);

	//cout << "processor " << rank << " ding ding ding " << endl;

	cout << " Area found in processor " << rank << " = "<< local_sum << endl;

  	//Terminate.

  	//Exiting MPI zone
  
  	MPI_Finalize();

  	//Trapezoidal rule Output
	
		
	if(rank == 0)
	{
		cout << endl << " Hermite Total Area = "<< global_sum << endl;
	}


  	return 0;

}



