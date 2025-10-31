#ifndef __SAVEFIELD_STRUCTS_H
#define __SAVEFIELD_STRUCTS_H

#include "main.cuh"

typedef struct saveField{
    int aux;
    bool checkpoint;
    bool save;
    bool reportSave;
    bool macrSave;
    bool particleSave;
   
    void flagsUpdate(int step) {
        aux = step - INI_STEP;
        checkpoint = false;
        save = false;
        reportSave = false;
        macrSave = false;
        particleSave = false;
    
        #pragma warning(push)
        #pragma warning(disable: 4804)
                if(aux != 0){
                    if(REPORT_SAVE){ reportSave = !(step % REPORT_SAVE);}                
                    if(MACR_SAVE){ macrSave   = !(step % MACR_SAVE);}
                    if(MACR_SAVE || REPORT_SAVE){ save = (reportSave || macrSave);}
                    if(CHECKPOINT_SAVE){ checkpoint = !(aux % CHECKPOINT_SAVE);}
                    #ifdef PARTICLE_MODEL
                        if(PARTICLES_SAVE){ particleSave = !(aux % PARTICLES_SAVE);}
                    #endif //PARTICLE MODEL
                }
        #pragma warning(pop)
    }
    

} SaveField;
#endif //__SAVEFIELD_STRUCTS_H