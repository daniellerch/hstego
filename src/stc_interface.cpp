#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstdlib>
#include <ctime>

#include "stc_embed_c.h"
#include "stc_extract_c.h"
#include "stc_interface.h"



int stc_hide(
    uint cover_length, 
    int* cover, 
    float* costs,
    uint message_length, 
    u8* message,
    int* stego
    ) {

    double dist;
    dist = stc_embed((u8*)cover, cover_length, message, message_length, (void*)costs, false, (u8*)stego);  

    return 0;
}


int stc_unhide(
    uint stego_length, 
    int* stego, 
    uint message_length, 
    u8* message
    ) {

    stc_extract((u8*)stego, stego_length, (u8*)message, message_length);

    return 0;
}


