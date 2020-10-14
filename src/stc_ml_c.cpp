#include "stc_ml_c.h"

#include <xmmintrin.h>
#include <cmath>
#include <limits>
#include <algorithm>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <string.h> // due to memcpy


#include <boost/random/uniform_int.hpp>       // this is required for Marsene-Twister random number generator
#include <boost/random/variate_generator.hpp>
#include <boost/random/mersenne_twister.hpp>


#include "stc_embed_c.h"
#include "stc_extract_c.h"
#include "sse_mathfun.h"    // library with optimized functions obtained from http://gruntthepeon.free.fr/ssemath/

// {{{ write_vector_to_file()
template< class T > void write_vector_to_file( uint n, T *ptr, const char* file_name ) {

    std::ofstream f( file_name );
    for ( uint i = 0; i < n; i++ )
        f << std::left << std::setw( 20 ) << i << std::left << std::setw( 20 ) << ptr[i] << std::endl;
    f.close();
}
// }}}

// {{{ write_matrix_to_file()
// write column-ordered matrix into file
template< class T > void write_matrix_to_file( uint rows, uint columns, T *ptr, const char* file_name ) {

    std::ofstream f( file_name );
    for ( uint i = 0; i < rows; i++ ) {
        f << std::left << std::setw( 20 ) << i;
        for ( uint j = 0; j < columns; j++ )
            f << std::left << std::setw( 20 ) << ptr[j * rows + i];
        f << std::endl;
    }
    f.close();
}
// }}}

// {{{ align_*()
// Templates to handle aligned version of new and delete operators.                                      
// These functions are necessary for creating arrays aligned address of certain multiples, such as 16. 
template< class T > T* align_new( unsigned int n, unsigned int align_size ) {
    char *ptr, *ptr2, *aligned_ptr;
    int align_mask = align_size - 1;

    ptr = new char[n * sizeof(T) + align_size + sizeof(int)];
    if ( ptr == 0 ) return 0;

    ptr2 = ptr + sizeof(int);
    aligned_ptr = ptr2 + (align_size - ((size_t) ptr2 & align_mask));

    ptr2 = aligned_ptr - sizeof(int);
    *((int*) ptr2) = (int) (aligned_ptr - ptr);

    return (T*) aligned_ptr;
}

template< class T > void align_delete( T *ptr ) {
    int *ptr2 = (int*) ptr - 1;
    char *p;

    p = (char*) ptr;
    p -= *ptr2;
    delete[] p;
}
// }}}

// {{{ randperm()
/* Generates random permutation of length n based on the MT random number generator with seed 'seed'. */
void randperm( uint n, uint seed, uint* perm ) {

    boost::mt19937 *generator = new boost::mt19937( seed );
    boost::variate_generator< boost::mt19937, boost::uniform_int< > > *randi = new boost::variate_generator< boost::mt19937,
        boost::uniform_int< > >( *generator, boost::uniform_int< >( 0, INT_MAX ) );

    // generate random permutation - this is used to shuffle cover pixels to randomize the effect of different neighboring pixels
    for ( uint i = 0; i < n; i++ )
        perm[i] = i;
    for ( uint i = 0; i < n; i++ ) {
        uint j = (*randi)() % (n - i);
        uint tmp = perm[i];
        perm[i] = perm[i + j];
        perm[i + j] = tmp;
    }

    delete generator;
    delete randi;
}
// }}}

// {{{ sum_inplace()
inline float sum_inplace( __m128 x ) {
    float y;
    // add all 4 terms from x together
    x = _mm_add_ps( x, _mm_shuffle_ps(x,x,_MM_SHUFFLE(1,0,3,2)) );
    x = _mm_add_ps( x, _mm_shuffle_ps(x,x,_MM_SHUFFLE(2,3,0,1)) );
    _mm_store_ss( &y, x );
    return y;
}
// }}}

// {{{ calc_entropy()
float calc_entropy( uint n, uint k, float* costs, float lambda ) {

    float const LOG2 = log( 2.0 );
    __m128 inf = _mm_set1_ps( F_INF );
    __m128 v_lambda = _mm_set1_ps( -lambda );
    __m128 z, d, rho, p, entr, mask;

    entr = _mm_setzero_ps();
    for ( uint i = 0; i < n / 4; i++ ) {
        z = _mm_setzero_ps();
        d = _mm_setzero_ps();
        for ( uint j = 0; j < k; j++ ) {
            rho = _mm_load_ps( costs + j * n + 4 * i ); // costs array must be aligned in memory
            p = exp_ps( _mm_mul_ps( v_lambda, rho ) );
            z = _mm_add_ps( z, p );

            mask = _mm_cmpeq_ps( rho, inf ); // if p<eps, then do not accumulate it to d since x*exp(-x) tends to zero
            p = _mm_mul_ps( rho, p );
            p = _mm_andnot_ps( mask, p ); // apply mask
            d = _mm_add_ps( d, p );
        }
        entr = _mm_sub_ps( entr, _mm_div_ps( _mm_mul_ps( v_lambda, d ), z ) );
        entr = _mm_add_ps( entr, log_ps( z ) );
    }
    return sum_inplace( entr ) / LOG2;
}
// }}}

// {{{ get_lambda_entropy()
float get_lambda_entropy( uint n, uint k, float *costs, float payload, float initial_lambda = 10 ) {

    float p1, p2, p3, lambda1, lambda2, lambda3;
    int j = 0;
    uint iterations = 0;

    lambda1 = 0;
    p1 = n * log( (float)k ) / log( 2.0f );
    lambda3 = initial_lambda;
    p3 = payload + 1; // this is just an initial value
    lambda2 = initial_lambda;
    while ( p3 > payload ) {
        lambda3 *= 2;
        p3 = calc_entropy( n, k, costs, lambda3 );
        j++;
        iterations++;
        // beta is probably unbounded => it seems that we cannot find beta such that
        // relative payload will be smaller than requested. Binary search does not make sence here.
        if ( j > 10 ) {
            return lambda3;
        }
    }
    while ( (p1 - p3) / n > payload / n * 1e-2 ) { // binary search for parameter lambda
        lambda2 = lambda1 + (lambda3 - lambda1) / 2;
        p2 = calc_entropy( n, k, costs, lambda2 );
        if ( p2 < payload ) {
            lambda3 = lambda2;
            p3 = p2;
        } else {
            lambda1 = lambda2;
            p1 = p2;
        }
        iterations++; // this is for monitoring the number of iterations
    }
    return lambda1 + (lambda3 - lambda1) / 2;
}
// }}}

// {{{ calc_distortion()
float calc_distortion( uint n, uint k, float* costs, float lambda ) {

    __m128 eps = _mm_set1_ps( std::numeric_limits< float >::epsilon() );
    __m128 v_lambda = _mm_set1_ps( -lambda );
    __m128 z, d, rho, p, dist, mask;

    dist = _mm_setzero_ps();
    for ( uint i = 0; i < n / 4; i++ ) { // n must be multiple of 4
        z = _mm_setzero_ps();
        d = _mm_setzero_ps();
        for ( uint j = 0; j < k; j++ ) {
            rho = _mm_load_ps( costs + j * n + 4 * i ); // costs array must be aligned in memory
            p = exp_ps( _mm_mul_ps( v_lambda, rho ) );
            z = _mm_add_ps( z, p );
            mask = _mm_cmplt_ps( p, eps ); // if p<eps, then do not accumulate it to d since x*exp(-x) tends to zero
            p = _mm_mul_ps( rho, p );
            p = _mm_andnot_ps( mask, p );
            d = _mm_add_ps( d, p );
        }
        dist = _mm_add_ps( dist, _mm_div_ps( d, z ) );
    }
    return sum_inplace( dist );
}
// }}}

// {{{ get_lambda_distortion()
float get_lambda_distortion( uint n, uint k, float *costs, float distortion, float initial_lambda = 10, float precision = 1e-3,
        uint iter_limit = 30 ) {

    float dist1, dist2, dist3, lambda1, lambda2, lambda3;
    int j = 0;
    uint iterations = 0;

    lambda1 = 0;
    dist1 = calc_distortion( n, k, costs, lambda1 );
    lambda3 = initial_lambda;
    dist2 = F_INF; // this is just an initial value
    lambda2 = initial_lambda;
    dist3 = distortion + 1;
    while ( dist3 > distortion ) {
        lambda3 *= 2;
        dist3 = calc_distortion( n, k, costs, lambda3 );
        j++;
        iterations++;
        // beta is probably unbounded => it seems that we cannot find beta such that
        // relative payload will be smaller than requested. Binary search cannot converge.
        if ( j > 10 ) {
            return lambda3;
        }
    }
    while ( (fabs( dist2 - distortion ) / n > precision) && (iterations < iter_limit) ) { // binary search for parameter lambda
        lambda2 = lambda1 + (lambda3 - lambda1) / 2;
        dist2 = calc_distortion( n, k, costs, lambda2 );
        if ( dist2 < distortion ) {
            lambda3 = lambda2;
            dist3 = dist2;
        } else {
            lambda1 = lambda2;
            dist1 = dist2;
        }
        iterations++; // this is for monitoring the number of iterations
    }
    return lambda1 + (lambda3 - lambda1) / 2;
}
// }}}

// {{{ binary_entropy_array()
float binary_entropy_array( uint n, float *prob ) {

    float h = 0;
    float const LOG2 = log( 2.0 );
    float const EPS = std::numeric_limits< float >::epsilon();

    for ( uint i = 0; i < n; i++ )
        if ( (prob[i] > EPS) && (1 - prob[i] > EPS) ) h -= prob[i] * log( prob[i] ) + (1 - prob[i]) * log( 1 - prob[i] );

    return h / LOG2;
}
// }}}

// {{{ entropy_array()
float entropy_array( uint n, float* prob ) {

    double h = 0;
    double const LOG2 = log( 2.0 );
    double const EPS = std::numeric_limits< double >::epsilon();

    for ( uint i = 0; i < n; i++ )
        if ( prob[i] > EPS ) h -= prob[i] * log( prob[i] );

    return h / LOG2;
}
// }}}

// {{{ mod()
inline uint mod( int x, int m ) {
    int tmp = x - (x / m) * m + m;
    return tmp % m;
}
// }}}



/* EMBEDDING ALGORITHMS */

// {{{ stc_embed_trial()
void stc_embed_trial( uint n, float* cover_bit_prob0, u8* message, uint stc_constraint_height, uint &num_msg_bits, uint* perm, u8* stego,
        uint &trial, uint max_trials, const char* debugging_file = "cost.txt" ) {

    bool success = false;
    u8* cover = new u8[n];
    double* cost = new double[n];
    while ( !success ) {
        randperm( n, num_msg_bits, perm );
        for ( uint i = 0; i < n; i++ ) {
            cover[perm[i]] = (cover_bit_prob0[i] < 0.5) ? 1 : 0;
            cost[perm[i]] = -log( (1 / std::max( cover_bit_prob0[i], 1 - cover_bit_prob0[i] )) - 1 );
            if ( cost[perm[i]] != cost[perm[i]] ) // if p20[i]>1 due to numerical error (this is possible due to float data type)
            cost[perm[i]] = D_INF; // then cost2[i] is NaN, it should be Inf
        }
        memcpy( stego, cover, n ); // initialize stego array by cover array
        // debugging
        // write_vector_to_file<double>(n, cost, debugging_file);
        try {
            if ( num_msg_bits != 0 ) stc_embed( cover, n, message, num_msg_bits, (void*) cost, true, stego, stc_constraint_height );
            success = true;
        } catch ( stc_exception& e ) {
            if ( e.error_id != 4 ) { // error_id=4 means No solution exists, thus we try to embed with different permutation.
                delete[] cost;
                delete[] cover;
                throw e;
            }
            num_msg_bits--; // by decreasing the number of  bits, we change the permutation used to shuffle the bits
            trial++;
            if ( trial > max_trials ) {
                delete[] cost;
                delete[] cover;
                throw stc_exception( "Maximum number of trials in layered construction exceeded (2).", 6 );
            }
        }
    }
    delete[] cost;
    delete[] cover;
}
// }}}

// {{{ check_costs()
// SANITY CHECKS for cost arrays
void check_costs( uint n, uint k, float *costs ) {

    bool test_nan, test_non_inf, test_minus_inf;
    for ( uint i = 0; i < n; i++ ) {
        test_nan = false; // Is any element NaN? Should be FALSE
        test_non_inf = false; // Is any element finite? Should be TRUE
        test_minus_inf = false; // Is any element minus Inf? should be FALSE
        for ( uint j = 0; j < k; j++ ) {
            test_nan |= (costs[k * i + j] != costs[k * i + j]);
            test_non_inf |= ((costs[k * i + j] != -F_INF) & (costs[k * i + j] != F_INF));
            test_minus_inf |= (costs[k * i + j] == -F_INF);
        }
        if ( test_nan ) {
            std::stringstream ss;
            ss << "Incorrect cost array." << i << "-th element contains NaN value. This is not a valid cost.";
            throw stc_exception( ss.str(), 6 );
        }
        if ( !test_non_inf ) {
            std::stringstream ss;
            ss << "Incorrect cost array." << i << "-th element does not contain any finite cost value. This is not a valid cost.";
            throw stc_exception( ss.str(), 6 );
        }
        if ( test_minus_inf ) {
            std::stringstream ss;
            ss << "Incorrect cost array." << i << "-th element contains -Inf value. This is not a valid cost.";
            throw stc_exception( ss.str(), 6 );
        }
    }
}
// }}}

// {{{ stc_pm1_pls_embed()
// MULTI-LAYERED EMBEDDING for plus/minus one changes
// payload limited case - returns distortion
float stc_pm1_pls_embed( uint cover_length, int* cover, float* costs, uint message_length, u8* message, // input variables
                         uint stc_constraint_height, float wet_cost,                                    // other input parameters
                         int* stego, uint* num_msg_bits, uint &max_trials, float* coding_loss ) {       // output variables

    return stc_pm1_dls_embed( cover_length, cover, costs, message_length, message, F_INF, stc_constraint_height, 0, wet_cost, stego,
            num_msg_bits, max_trials, coding_loss );
}
// }}}

// {{{ stc_pm1_dls_embed()
// distortion limited case - returns distortion
float stc_pm1_dls_embed( uint cover_length, int* cover, float* costs, uint message_length, u8* message, float target_distortion, // input variables
                         uint stc_constraint_height, float expected_coding_loss, float wet_cost,   // other input parameters
                         int* stego, uint* num_msg_bits, uint &max_trials, float* coding_loss ) {  // output variables

    check_costs( cover_length, 3, costs );
    float dist = 0;

    int *stego_values = new int[4 * cover_length];
    float *costs_ml2 = new float[4 * cover_length];
    for ( uint i = 0; i < cover_length; i++ ) {
        costs_ml2[4 * i + mod( (cover[i] - 1 + 4), 4 )] = costs[3 * i + 0]; // set cost of changing by -1
        stego_values[4 * i + mod( (cover[i] - 1 + 4), 4 )] = cover[i] - 1;
        costs_ml2[4 * i + mod( (cover[i] + 0 + 4), 4 )] = costs[3 * i + 1]; // set cost of changing by 0
        stego_values[4 * i + mod( (cover[i] + 0 + 4), 4 )] = cover[i];
        costs_ml2[4 * i + mod( (cover[i] + 1 + 4), 4 )] = costs[3 * i + 2]; // set cost of changing by +1
        stego_values[4 * i + mod( (cover[i] + 1 + 4), 4 )] = cover[i] + 1;
        costs_ml2[4 * i + mod( (cover[i] + 2 + 4), 4 )] = wet_cost; // set cost of changing by +2
        stego_values[4 * i + mod( (cover[i] + 2 + 4), 4 )] = cover[i] + 2;
    }

    // run general 2 layered embedding in distortion limited regime
    dist = stc_ml2_embed( cover_length, costs_ml2, stego_values, message_length, message, target_distortion, stc_constraint_height,
            expected_coding_loss, stego, num_msg_bits, max_trials, coding_loss );
    delete[] costs_ml2;
    delete[] stego_values;

    return dist;
}
// }}}

// {{{ stc_pm2_dls_embed()
// MULTI-LAYERED EMBEDDING for plus/minus one and two changes
// payload limited case - returns distortion
float stc_pm2_pls_embed( uint cover_length, int* cover, float* costs, uint message_length, u8* message, // input variables
        uint stc_constraint_height, float wet_cost, // other input parameters
        int* stego, uint* num_msg_bits, uint &max_trials, float* coding_loss ) { // output variables

    return stc_pm2_dls_embed( cover_length, cover, costs, message_length, message, F_INF, stc_constraint_height, 0, wet_cost, stego,
            num_msg_bits, max_trials, coding_loss );
}
// }}}

// {{{ stc_pm2_dls_embed()
// distortion limited case - returns distortion
float stc_pm2_dls_embed( uint cover_length, int* cover, float* costs, uint message_length, u8* message, float target_distortion, // input variables
        uint stc_constraint_height, float expected_coding_loss, float wet_cost, // other input parameters
        int* stego, uint* num_msg_bits, uint &max_trials, float* coding_loss ) { // output variables

    check_costs( cover_length, 5, costs );
    int *stego_values = new int[8 * cover_length];
    float* costs_ml3 = new float[8 * cover_length];
    std::fill_n( costs_ml3, 8 * cover_length, wet_cost ); // initialize new cost array

    for ( uint i = 0; i < cover_length; i++ ) {
        costs_ml3[8 * i + mod( (cover[i] - 2 + 8), 8 )] = costs[5 * i + 0]; // set cost of changing by -2
        stego_values[8 * i + mod( (cover[i] - 2 + 8), 8 )] = cover[i] - 2;
        costs_ml3[8 * i + mod( (cover[i] - 1 + 8), 8 )] = costs[5 * i + 1]; // set cost of changing by -1
        stego_values[8 * i + mod( (cover[i] - 1 + 8), 8 )] = cover[i] - 1;
        costs_ml3[8 * i + mod( (cover[i] + 0 + 8), 8 )] = costs[5 * i + 2]; // set cost of changing by 0
        stego_values[8 * i + mod( (cover[i] + 0 + 8), 8 )] = cover[i] + 0;
        costs_ml3[8 * i + mod( (cover[i] + 1 + 8), 8 )] = costs[5 * i + 3]; // set cost of changing by +1
        stego_values[8 * i + mod( (cover[i] + 1 + 8), 8 )] = cover[i] + 1;
        costs_ml3[8 * i + mod( (cover[i] + 2 + 8), 8 )] = costs[5 * i + 4]; // set cost of changing by +2
        stego_values[8 * i + mod( (cover[i] + 2 + 8), 8 )] = cover[i] + 2;
        stego_values[8 * i + mod( (cover[i] + 3 + 8), 8 )] = cover[i] + 3; // these values are not used and are defined
        stego_values[8 * i + mod( (cover[i] + 4 + 8), 8 )] = cover[i] + 4; // just to have the array complete
        stego_values[8 * i + mod( (cover[i] + 5 + 8), 8 )] = cover[i] + 5; //
    }

    // run general 3 layered embedding in distortion limited regime
    float dist = stc_ml3_embed( cover_length, costs_ml3, stego_values, message_length, message, target_distortion, stc_constraint_height,
            expected_coding_loss, stego, num_msg_bits, max_trials, coding_loss );
    delete[] costs_ml3;
    delete[] stego_values;

    return dist;
}
// }}}

// GENERAL MULTI-LAYERED EMBEDDING

// {{{ stc_ml1_embed()
// algorithm for embedding into 1 layer, both payload- and distortion-limited case
float stc_ml1_embed( uint cover_length, int* cover, short* direction, float* costs, uint message_length, u8* message,
        float target_distortion,// input variables
        uint stc_constraint_height, float expected_coding_loss, // other input parameters
        int* stego, uint* num_msg_bits, uint &max_trials, float* coding_loss ) { // output variables

    float distortion, lambda = 0, m_max = 0;
    bool success = false;
    uint m_actual = 0;
    uint n = cover_length + 4 - (cover_length % 4); // cover length rounded to multiple of 4
    uint *perm1 = new uint[n];

    float* c = align_new< float > ( 2 * n, 16 );
    std::fill_n( c, 2 * n, F_INF );
    std::fill_n( c, n, 0 );
    for ( uint i = 0; i < cover_length; i++ ) { // copy and transpose data for better reading via SSE instructions
        c[mod( cover[i], 2 ) * n + i] = 0; // cost of not changing the element
        c[mod( (cover[i] + 1), 2 ) * n + i] = costs[i]; // cost of changing the element
    }

    if ( target_distortion != F_INF ) { // distortion-limited sender
        lambda = get_lambda_distortion( n, 2, c, target_distortion, 2 ); //
        m_max = (1 - expected_coding_loss) * calc_entropy( n, 2, c, lambda ); //
        m_actual = std::min( message_length, (uint) floor( m_max ) ); //
    }
    if ( (target_distortion == F_INF) || (m_actual < floor( m_max )) ) { // payload-limited sender
        m_actual = std::min( cover_length, message_length ); // or distortion-limited sender with
    }

    /* SINGLE LAYER OF 1ST LSBs */
    num_msg_bits[0] = m_actual;
    uint trial = 0;
    u8* cover1 = new u8[cover_length];
    double* cost1 = new double[cover_length];
    u8* stego1 = new u8[cover_length];
    while ( !success ) {
        randperm( cover_length, num_msg_bits[0], perm1 );
        for ( uint i = 0; i < cover_length; i++ ) {
            cover1[perm1[i]] = mod( cover[i], 2 );
            cost1[perm1[i]] = costs[i];
            if ( cost1[perm1[i]] != cost1[perm1[i]] ) cost1[perm1[i]] = D_INF;
        }
        memcpy( stego1, cover1, cover_length ); // initialize stego array by cover array
        // debugging
        // write_vector_to_file<double>(n, cost, debugging_file);
        try {
            if ( num_msg_bits[0] != 0 ) stc_embed( cover1, cover_length, message, num_msg_bits[0], (void*) cost1, true, stego1,
                    stc_constraint_height );
            success = true;
        } catch ( stc_exception& e ) {
            if ( e.error_id != 4 ) { // error_id=4 means No solution exists, thus we try to embed with different permutation.
                delete[] cost1;
                delete[] cover1;
                delete[] stego1;
                delete[] perm1;
                delete[] c;
                throw e;
            }
            num_msg_bits[0]--; // by decreasing the number of  bits, we change the permutation used to shuffle the bits
            trial++;
            if ( trial > max_trials ) {
                delete[] cost1;
                delete[] cover1;
                delete[] stego1;
                delete[] perm1;
                delete[] c;
                throw stc_exception( "Maximum number of trials in layered construction exceeded (1).", 6 );
            }
        }
    }

    /* FINAL CALCULATIONS */
    distortion = 0;
    for ( uint i = 0; i < cover_length; i++ ) {
        stego[i] = (stego1[perm1[i]] == cover1[perm1[i]]) ? cover[i] : cover[i] + direction[i];
        distortion += (stego1[perm1[i]] == cover1[perm1[i]]) ? 0 : costs[i];
    }
    if ( coding_loss != 0 ) {
        float lambda_dist = get_lambda_distortion( n, 2, c, distortion, lambda, 0, 20 ); // use 20 iterations to make lambda_dist precise
        float max_payload = calc_entropy( n, 2, c, lambda_dist );
        (*coding_loss) = (max_payload - m_actual) / max_payload; // fraction of max_payload lost due to practical coding scheme
    }
    max_trials = trial;

    delete[] cost1;
    delete[] cover1;
    delete[] stego1;
    delete[] perm1;
    align_delete< float > ( c );

    return distortion;
}
// }}}

// {{{ stc_ml2_embed()
// algorithm for embedding into 2 layers with possibility to use only 1 layer, both payload- and distortion-limited cases
float stc_ml2_embed( uint cover_length, float* costs, int* stego_values, uint message_length, u8* message, float target_distortion, // input variables
        uint stc_constraint_height, float expected_coding_loss, // other input parameters
        int* stego, uint* num_msg_bits, uint &max_trials, float* coding_loss ) { // output and optional variables

    float distortion, dist_coding_loss, lambda = 0, m_max = 0;
    uint m_actual = 0;
    uint n = cover_length + 4 - (cover_length % 4); // cover length rounded to multiple of 4

    check_costs( cover_length, 4, costs );
    // if only binary embedding is sufficient, then use only 1st LSB layer
    bool lsb1_only = true;
    for ( uint i = 0; i < cover_length; i++ ) {
        uint n_finite_costs = 0; // number of finite cost values
        uint lsb_xor = 0;
        for ( uint k = 0; k < 4; k++ )
            if ( costs[4 * i + k] != F_INF ) {
                n_finite_costs++;
                lsb_xor ^= (k % 2);
            }
        lsb1_only &= ((n_finite_costs <= 2) & (lsb_xor == 1));
    }
    if ( lsb1_only ) { // use stc_ml1_embed method
        distortion = 0;
        int *cover = new int[cover_length];
        short *direction = new short[cover_length];
        float *costs_ml1 = new float[cover_length];
        for ( uint i = 0; i < cover_length; i++ ) { // normalize such that minimal element is 0 - this helps numerical stability
            uint min_id = 0;
            float f_min = F_INF;
            for ( uint j = 0; j < 4; j++ )
                if ( f_min > costs[4 * i + j] ) {
                    f_min = costs[4 * i + j]; // minimum value
                    min_id = j; // index of the minimal entry
                }
            costs_ml1[i] = F_INF;
            cover[i] = stego_values[4 * i + min_id];
            for ( uint j = 0; j < 4; j++ )
                if ( (costs[4 * i + j] != F_INF) && (min_id != j) ) {
                    distortion += f_min;
                    costs_ml1[i] = costs[4 * i + j] - f_min;
                    direction[i] = stego_values[4 * i + j] - cover[i];
                }
        }

        distortion += stc_ml1_embed( cover_length, cover, direction, costs_ml1, message_length, message, target_distortion,
                stc_constraint_height, expected_coding_loss, stego, num_msg_bits, max_trials, coding_loss );
        delete[] direction;
        delete[] costs_ml1;
        delete[] cover;
        return distortion;
    }

    // copy and transpose data for faster reading via SSE instructions
    float* c = align_new< float > ( 4 * n, 16 );
    std::fill_n( c, 4 * n, F_INF );
    std::fill_n( c, n, 0 );
    for ( uint i = 0; i < 4 * cover_length; i++ )
        c[n * (i % 4) + i / 4] = costs[i];
    // write_matrix_to_file<float>(n, 4, c, "cost_ml2.txt");
    for ( uint i = 0; i < n; i++ ) { // normalize such that minimal element is 0 - this helps numerical stability
        float f_min = F_INF;
        for ( uint j = 0; j < 4; j++ )
            f_min = std::min( f_min, c[j * n + i] );
        for ( uint j = 0; j < 4; j++ )
            c[j * n + i] -= f_min;
    }

    if ( target_distortion != F_INF ) {
        lambda = get_lambda_distortion( n, 4, c, target_distortion, 2 );
        m_max = (1 - expected_coding_loss) * calc_entropy( n, 4, c, lambda );
        m_actual = std::min( message_length, (uint) floor( m_max ) );
    }
    if ( (target_distortion == F_INF) || (m_actual < floor( m_max )) ) {
        m_actual = std::min( 2 * cover_length, message_length );
        lambda = get_lambda_entropy( n, 4, c, m_actual, 2 );
    }
    /* 
     p = exp(-lambda*costs);
     p = p./(ones(4,1)*sum(p));
     */
    float* p = align_new< float > ( 4 * n, 16 );
    __m128 v_lambda = _mm_set1_ps( -lambda );
    for ( uint i = 0; i < n / 4; i++ ) {
        __m128 sum = _mm_setzero_ps();
        for ( uint j = 0; j < 4; j++ ) {
            __m128 x = _mm_load_ps( c + j * n + 4 * i );
            x = exp_ps( _mm_mul_ps( v_lambda, x ) );
            _mm_store_ps( p + j * n + 4 * i, x );
            sum = _mm_add_ps( sum, x );
        }
        for ( uint j = 0; j < 4; j++ ) {
            __m128 x = _mm_load_ps( p + j * n + 4 * i );
            x = _mm_div_ps( x, sum );
            _mm_store_ps( p + j * n + 4 * i, x );
        }
    }
    // this is for debugging purposes
    // float payload_dbg = entropy_array(4*n, p);

    uint trial = 0;
    float* p10 = new float[cover_length];
    float* p20 = new float[cover_length];
    u8* stego1 = new u8[cover_length];
    u8* stego2 = new u8[cover_length];
    uint *perm1 = new uint[cover_length];
    uint *perm2 = new uint[cover_length];

    /* LAYER OF 2ND LSBs */
    for ( uint i = 0; i < cover_length; i++ )
        p20[i] = p[i] + p[i + n]; // p20 = p(1,:)+p(2,:);         % probability of 2nd LSB of stego equal 0
    //num_msg_bits[1] = (uint) floor( binary_entropy_array( cover_length, p20 ) ); // msg_bits(2) = floor(sum(binary_entropy(p20)));    % number of msg bits embedded into 2nd LSBs
    num_msg_bits[1] = (uint) (message_length/2 /*+ message_length%2*/ ); // XXX

    try {
        stc_embed_trial( cover_length, p20, message, stc_constraint_height, num_msg_bits[1], perm2, stego2, trial, max_trials, "cost2.txt" );
    } catch ( stc_exception& e ) {
        delete[] p10;
        delete[] p20;
        delete[] perm1;
        delete[] perm2;
        delete[] stego1;
        delete[] stego2;
        align_delete< float > ( c );
        align_delete< float > ( p );
        throw e;
    }

    /* LAYER OF 1ST LSBs */
    for ( uint i = 0; i < cover_length; i++ ) //
        if ( stego2[perm2[i]] == 0 ) // % conditional probability of 1st LSB of stego equal 0 given LSB2=0
        p10[i] = p[i] / (p[i] + p[i + n]); // p10(i) = p(1,i)/(p(1,i)+p(2,i));
        else // % conditional probability of 1st LSB of stego equal 0 given LSB2=1
        p10[i] = p[i + 2 * n] / (p[i + 2 * n] + p[i + 3 * n]); // p10(i) = p(3,i)/(p(3,i)+p(4,i));
    num_msg_bits[0] = m_actual - num_msg_bits[1]; // msg_bits(1) = m_actual-msg_bits(2); % number of msg bits embedded into 1st LSBs
    try {
        stc_embed_trial( cover_length, p10, message + num_msg_bits[1], stc_constraint_height, num_msg_bits[0], perm1, stego1, trial,
                max_trials, "cost1.txt" );
    } catch ( stc_exception& e ) {
        delete[] p10;
        delete[] p20;
        delete[] perm1;
        delete[] perm2;
        delete[] stego1;
        delete[] stego2;
        align_delete< float > ( c );
        align_delete< float > ( p );
        throw e;
    }
    delete[] p10;
    delete[] p20;

    /* FINAL CALCULATIONS */
    distortion = 0;
    for ( uint i = 0; i < cover_length; i++ ) {
        stego[i] = stego_values[4 * i + 2 * stego2[perm2[i]] + stego1[perm1[i]]];
        distortion += costs[4 * i + 2 * stego2[perm2[i]] + stego1[perm1[i]]];
    }
    if ( coding_loss != 0 ) {
        dist_coding_loss = 0;
        for ( uint i = 0; i < cover_length; i++ )
            dist_coding_loss += c[i + n * (2 * stego2[perm2[i]] + stego1[perm1[i]])];
        float lambda_dist = get_lambda_distortion( n, 4, c, dist_coding_loss, lambda, 0, 20 ); // use 20 iterations to make lambda_dist precise
        float max_payload = calc_entropy( n, 4, c, lambda_dist );
        (*coding_loss) = (max_payload - m_actual) / max_payload; // fraction of max_payload lost due to practical coding scheme
    }
    max_trials = trial;

    delete[] stego1;
    delete[] stego2;
    delete[] perm1;
    delete[] perm2;
    align_delete< float > ( c );
    align_delete< float > ( p );

    return distortion;
}
// }}}

// {{{ stc_ml3_embed()
// algorithm for embedding into 3 layers, both payload- and distortion-limited case
float stc_ml3_embed( uint cover_length, float* costs, int* stego_values, uint message_length, u8* message, float target_distortion, // input variables
        uint stc_constraint_height, float expected_coding_loss, // other input parameters
        int* stego, uint* num_msg_bits, uint &max_trials, float* coding_loss ) { // output and optional variables

    float distortion, dist_coding_loss, lambda = 0, m_max = 0;
    uint m_actual = 0;
    uint n = cover_length + 4 - (cover_length % 4); // cover length rounded to multiple of 4

    check_costs( cover_length, 8, costs );
    float* c = align_new< float > ( 8 * n, 16 );
    std::fill_n( c, 8 * n, F_INF );
    std::fill_n( c, n, 0 );
    for ( uint i = 0; i < 8 * cover_length; i++ )
        c[n * (i % 8) + i / 8] = costs[i]; // copy and transpose data for better reading via SSE instructions
    // write_matrix_to_file<float>(n, 8, c, "cost_ml3.txt");
    for ( uint i = 0; i < n; i++ ) { // normalize such that minimal element is 0 - this helps numerical stability
        float f_min = F_INF;
        for ( uint j = 0; j < 8; j++ )
            f_min = std::min( f_min, c[j * n + i] );
        for ( uint j = 0; j < 8; j++ )
            c[j * n + i] -= f_min;
    }

    if ( target_distortion != F_INF ) {
        lambda = get_lambda_distortion( n, 8, c, target_distortion, 2.0 );
        m_max = (1 - expected_coding_loss) * calc_entropy( n, 8, c, lambda );
        m_actual = std::min( message_length, (uint) floor( m_max ) );
    }
    if ( (target_distortion == F_INF) || (m_actual < floor( m_max )) ) {
        m_actual = std::min( 3 * cover_length, message_length );
        lambda = get_lambda_entropy( n, 8, c, m_actual, 2.0 );
    }
    /* 
     p = exp(-lambda*costs);
     p = p./(ones(8,1)*sum(p));
     */
    float* p = align_new< float > ( 8 * n, 16 );
    __m128 v_lambda = _mm_set1_ps( -lambda );
    for ( uint i = 0; i < n / 4; i++ ) {
        __m128 sum = _mm_setzero_ps();
        for ( uint j = 0; j < 8; j++ ) {
            __m128 x = _mm_load_ps( c + j * n + 4 * i );
            x = exp_ps( _mm_mul_ps( v_lambda, x ) );
            _mm_store_ps( p + j * n + 4 * i, x );
            sum = _mm_add_ps( sum, x );
        }
        for ( uint j = 0; j < 8; j++ ) {
            __m128 x = _mm_load_ps( p + j * n + 4 * i );
            x = _mm_div_ps( x, sum );
            _mm_store_ps( p + j * n + 4 * i, x );
        }
    }
    // this is for debugging
    // float payload_dbg = entropy_array(8*n, p);

    uint trial = 0;
    float* p10 = new float[cover_length];
    float* p20 = new float[cover_length];
    float* p30 = new float[cover_length];
    u8* stego1 = new u8[cover_length];
    u8* stego2 = new u8[cover_length];
    u8* stego3 = new u8[cover_length];
    uint *perm1 = new uint[cover_length];
    uint *perm2 = new uint[cover_length];
    uint *perm3 = new uint[cover_length];

    /* LAYER OF 3RD LSBs */
    for ( uint i = 0; i < cover_length; i++ )
        p30[i] = p[i] + p[i + n] + p[i + 2 * n] + p[i + 3 * n]; //
    num_msg_bits[2] = (uint) floor( binary_entropy_array( cover_length, p30 ) ); //
    try {
        stc_embed_trial( cover_length, p30, message, stc_constraint_height, num_msg_bits[2], perm3, stego3, trial, max_trials, "cost3.txt" );
    } catch ( stc_exception& e ) {
        delete[] p10;
        delete[] p20;
        delete[] p30;
        delete[] perm1;
        delete[] perm2;
        delete[] perm3;
        delete[] stego1;
        delete[] stego2;
        delete[] stego3;
        align_delete< float > ( c );
        align_delete< float > ( p );
        throw e;
    }

    /* LAYER OF 2ND LSBs */
    for ( uint i = 0; i < cover_length; i++ ) { //
        int s = 4 * stego3[perm3[i]]; // % conditional probability of 2nd LSB of stego equal 0 given LSB3
        p20[i] = (p[i + s * n] + p[i + (s + 1) * n]) / (p[i + s * n] + p[i + (s + 1) * n] + p[i + (s + 2) * n] + p[i + (s + 3) * n]);
    }
    num_msg_bits[1] = (uint) floor( binary_entropy_array( cover_length, p20 ) );// msg_bits(2) = floor(sum(binary_entropy(p20)));    % number of msg bits embedded into 2nd LSBs
    try {
        stc_embed_trial( cover_length, p20, message + num_msg_bits[2], stc_constraint_height, num_msg_bits[1], perm2, stego2, trial,
                max_trials, "cost2.txt" );
    } catch ( stc_exception& e ) {
        delete[] p10;
        delete[] p20;
        delete[] p30;
        delete[] perm1;
        delete[] perm2;
        delete[] perm3;
        delete[] stego1;
        delete[] stego2;
        delete[] stego3;
        align_delete< float > ( c );
        align_delete< float > ( p );
        throw e;
    }

    /* LAYER OF 1ST LSBs */
    for ( uint i = 0; i < cover_length; i++ ) { //
        int s = 4 * stego3[perm3[i]] + 2 * stego2[perm2[i]]; // % conditional probability of 1st LSB of stego equal 0 given LSB3 and LSB2
        p10[i] = p[i + s * n] / (p[i + s * n] + p[i + (s + 1) * n]);
    }
    num_msg_bits[0] = m_actual - num_msg_bits[1] - num_msg_bits[2]; // msg_bits(1) = m_actual-msg_bits(2)-msg_bits(3); % number of msg bits embedded into 1st LSBs
    try {
        stc_embed_trial( cover_length, p10, message + num_msg_bits[1] + num_msg_bits[2], stc_constraint_height, num_msg_bits[0], perm1,
                stego1, trial, max_trials, "cost1.txt" );
    } catch ( stc_exception& e ) {
        delete[] p10;
        delete[] p20;
        delete[] p30;
        delete[] perm1;
        delete[] perm2;
        delete[] perm3;
        delete[] stego1;
        delete[] stego2;
        delete[] stego3;
        align_delete< float > ( c );
        align_delete< float > ( p );
        throw e;
    }
    delete[] p10;
    delete[] p20;
    delete[] p30;
    max_trials = trial;

    /* FINAL CALCULATIONS */
    distortion = 0;
    for ( uint i = 0; i < cover_length; i++ ) {
        stego[i] = stego_values[8 * i + 4 * stego3[perm3[i]] + 2 * stego2[perm2[i]] + stego1[perm1[i]]];
        distortion += costs[8 * i + 4 * stego3[perm3[i]] + 2 * stego2[perm2[i]] + stego1[perm1[i]]];
    }
    if ( coding_loss != 0 ) {
        dist_coding_loss = 0;
        for ( uint i = 0; i < cover_length; i++ )
            dist_coding_loss += c[i + n * (4 * stego3[perm3[i]] + 2 * stego2[perm2[i]] + stego1[perm1[i]])];
        float lambda_dist = get_lambda_distortion( n, 8, c, dist_coding_loss, lambda, 0, 20 ); // use 20 iterations to make lambda_dist precise
        float max_payload = calc_entropy( n, 8, c, lambda_dist );
        (*coding_loss) = (max_payload - m_actual) / max_payload; // fraction of max_payload lost due to practical coding scheme
    }

    delete[] perm1;
    delete[] perm2;
    delete[] perm3;
    delete[] stego1;
    delete[] stego2;
    delete[] stego3;
    align_delete< float > ( c );
    align_delete< float > ( p );

    return distortion;
}
// }}}


/* EXTRACTION ALGORITHMS */

// {{{ stc_ml_extract()
/** Extraction algorithm for any l-layered construction.
 @param stego_length - ...
 @param stego - ...
 @param msg_bits - ...
 @param stc_constraint_height - ...
 @param message - ...
 */
void stc_ml_extract( uint stego_length, int* stego, uint num_of_layers, uint* num_msg_bits, // input variables
                     uint stc_constraint_height, // other input parameters
                     u8* message ) { // output variables

    u8* stego_bits = new u8[stego_length];
    u8* msg_ptr = message;
    uint *perm = new uint[stego_length];

    for ( uint l = num_of_layers; l > 0; l-- ) { // extract message from every layer starting from most significant ones
        // extract bits from l-th LSB plane
        if ( num_msg_bits[l - 1] > 0 ) {
            randperm( stego_length, num_msg_bits[l - 1], perm );
            for ( uint i = 0; i < stego_length; i++ )
                stego_bits[perm[i]] = mod( stego[i], (1 << l) ) >> (l - 1);
            stc_extract( stego_bits, stego_length, msg_ptr, num_msg_bits[l - 1], stc_constraint_height );
            msg_ptr += num_msg_bits[l - 1];
        }
    }

    delete[] stego_bits;
    delete[] perm;
}
// }}}


