#include <cstdlib>
#include <ctime> 
#include <iostream>

#define number_of_tosses 1000


int main(){
    srand( time(nullptr) );
    int number_in_circle = 0;
    for ( int toss = 0; toss < number_of_tosses; toss ++) {
        double x =  ((double) rand()) / RAND_MAX;
        double y =  ((double) rand()) / RAND_MAX;
        double distance_squared = (x * x) + (y * y);
        if ( distance_squared <= 1)
            number_in_circle++;
    }
    double pi_estimate = 4 * number_in_circle /(( double ) number_of_tosses);
    std::cout << "Estimate of pi is " << pi_estimate << std::endl;
}