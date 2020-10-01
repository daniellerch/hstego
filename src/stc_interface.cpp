#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstdlib>
#include <ctime>

#include "stc_ml_c.h"
#include "stc_interface.h"


uint h = 10;    // constraint height of STC code

int stc_hide(
    uint cover_length, 
    int* cover, 
    float* costs,
    uint message_length, 
    u8* message,
    int* stego
    ) {

    const uint n = cover_length;       
    uint m = message_length;
    uint trials = 10;          // if the message cannot be embedded due to large amount of 
                               // wet pixels, then try again with smaller message. Try at most 10 times.
    
    // srand(time(NULL));


    unsigned int* num_msg_bits = new unsigned int[2];
    float coding_loss; // calculate coding loss
    float dist = stc_pm1_pls_embed(n, cover, costs, m, message,
                                   h, F_INF,
                                   stego, num_msg_bits, trials, &coding_loss); // trials contain the number of trials used
    delete[] num_msg_bits;

    return 0;
}


int stc_unhide(
    uint stego_length, 
    int* stego, 
    uint message_length, 
    u8* message
    ) {

    unsigned int* num_msg_bits = new unsigned int[2];
    num_msg_bits[0] = (uint)(message_length/3);
    num_msg_bits[1] = num_msg_bits[0];
    stc_ml_extract(stego_length, stego, 2, num_msg_bits, h, message);
    
    return 0;
}


