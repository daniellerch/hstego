
#ifndef STC_INTERFACE_H
#define STC_INTERFACE_H

extern "C" {
   int stc_hide(uint cover_length, int* cover, float* costs,
                uint message_length, u8* message, int* stego);

   int stc_unhide(uint stego_length, int* stego, 
                  uint message_length, u8* message);
}

#endif
