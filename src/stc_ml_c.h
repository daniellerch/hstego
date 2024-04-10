#ifndef STC_ML_H
#define STC_ML_H

#include <limits>
#include "common.h"
#include "stc_embed_c.h"
#include "stc_extract_c.h"

typedef unsigned int uint;
typedef unsigned char u8;

const float F_INF = std::numeric_limits<float>::infinity();
const float D_INF = std::numeric_limits<double>::infinity();

// EMBEDDING ALGORITHMS ***********************************************************************************************************

// MULTI-LAYERED EMBEDDING for plus/minus one changes
// payload limited case - returns distortion
float stc_pm1_pls_embed(uint cover_length, int* cover, float* costs, uint message_length, u8* message,                             // input variables
                        uint stc_constraint_height, float wet_cost,                                                                // other input parameters
                        int* stego, uint* num_msg_bits, uint &max_trials, float* coding_loss = 0);                                                   // output variables
// distortion limited case - returns distortion
float stc_pm1_dls_embed(uint cover_length, int* cover, float* costs, uint message_length, u8* message, float target_distortion,    // input variables
                        uint stc_constraint_height, float expected_coding_loss, float wet_cost,                                    // other input parameters
                        int* stego, uint* num_msg_bits, uint &max_trials, float* coding_loss = 0);                                                   // output variables

// MULTI-LAYERED EMBEDDING for plus/minus one and two changes
// payload limited case - returns distortion
float stc_pm2_pls_embed(uint cover_length, int* cover, float* costs, uint message_length, u8* message,                             // input variables
                        uint stc_constraint_height, float wet_cost,                                                                // other input parameters
                        int* stego, uint* num_msg_bits, uint &max_trials, float* coding_loss = 0);                                                   // output variables
// distortion limited case - returns distortion
float stc_pm2_dls_embed(uint cover_length, int* cover, float* costs, uint message_length, u8* message, float target_distortion,    // input variables
                        uint stc_constraint_height, float expected_coding_loss, float wet_cost,                                    // other input parameters
                        int* stego, uint* num_msg_bits, uint &max_trials, float* coding_loss = 0);                                                   // output variables

// GENERAL MULTI-LAYERED EMBEDDING
// algorithm for embedding into 1 layer, both payload- and distortion-limited case
float stc_ml1_embed(uint cover_length, int* cover, short* direction, float* costs, uint message_length, u8* message, float target_distortion,// input variables
                    uint stc_constraint_height, float expected_coding_loss,                                                        // other input parameters
                    int* stego, uint* num_msg_bits, uint &max_trials, float* coding_loss = 0);                                     // output variables
// algorithm for embedding into 2 layers, both payload- and distortion-limited case
float stc_ml2_embed(uint cover_length, float* costs, int* stego_values, uint message_length, u8* message, float target_distortion,        // input variables
                    uint stc_constraint_height, float expected_coding_loss,                                                        // other input parameters
                    int* stego, uint* num_msg_bits, uint &max_trials, float* coding_loss = 0);                                     // output and optional variables
// algorithm for embedding into 3 layers, both payload- and distortion-limited case
float stc_ml3_embed(uint cover_length, float* costs, int* stego_values, uint message_length, u8* message, float target_distortion,        // input variables
                    uint stc_constraint_height, float expected_coding_loss,                                                        // other input parameters
                    int* stego, uint* num_msg_bits, uint &max_trials, float* coding_loss = 0);                                     // output and optional variables

// EXTRACTION ALGORITHMS **********************************************************************************************************

/** Extraction algorithm for 2 layered construction. Can be used with: stc_pm1_pls_embed, stc_pm1_dls_embed, stc_ml2_embed
    @param stego_length - ...
    @param stego - ...
    @param msg_bits - ...
    @param stc_constraint_height - ...
    @param message - ...
*/
void stc_ml_extract(uint stego_length, int* stego, uint num_of_layers, uint* num_msg_bits, // input variables
                    uint stc_constraint_height,                                            // other input parameters
                    u8* message);                                                          // output variables

#endif // STC_ML_H
