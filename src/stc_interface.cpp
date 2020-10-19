#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstdlib>
#include <ctime>

#include "stc_ml_c.h"
#include "stc_interface.h"


uint h = 10;    // constraint height of STC code

int stc_hide(uint cover_length, int* cover, float* costs, 
             uint message_length, u8* message, int* stego) {

    const uint n = cover_length;       
    uint m = message_length;

    // if the message cannot be embedded due to large amount of 
    // wet pixels, then try again with smaller message. Try at most 10 times.
    uint trials = 10;

    //std::cout << "message_length: " << message_length << std::endl;
    unsigned int* num_msg_bits = new unsigned int[2];
    float dist = stc_pm1_pls_embed(n, cover, costs, m, message, h, 2147483647, stego, num_msg_bits, trials, 0); 
    //std::cout << "hide -->" << num_msg_bits[0] << ", " << num_msg_bits[1] << std::endl;
    delete[] num_msg_bits;

    return 0;
}


int stc_unhide(uint stego_length, int* stego, 
                  uint message_length, u8* message) {

    unsigned int* num_msg_bits = new unsigned int[2];
    num_msg_bits[1] = (uint) (message_length/2); 
    num_msg_bits[0] = message_length-num_msg_bits[1];

    //std::cout << "message_length: " << message_length << std::endl;
    //std::cout << "unhide -->" << num_msg_bits[0] << ", " << num_msg_bits[1] << std::endl;

    stc_ml_extract(stego_length, stego, 2, num_msg_bits, h, message);

    return 0;
}


