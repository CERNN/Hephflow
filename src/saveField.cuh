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
   
    /** @brief Updates save and checkpoint flags for the current simulation step. */
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
                    constexpr int reportInterval = static_cast<int>(REPORT_SAVE);
                    constexpr int macroInterval = static_cast<int>(MACR_SAVE);
                    constexpr int checkpointInterval = static_cast<int>(CHECKPOINT_SAVE);
                    if constexpr (reportInterval > 0) reportSave = !(step % reportInterval);
                    if constexpr (macroInterval > 0) macrSave = !(step % macroInterval);
                    if constexpr (macroInterval > 0 || reportInterval > 0) save = reportSave || macrSave;
                    if constexpr (checkpointInterval > 0) checkpoint = !(aux % checkpointInterval);
                    #ifdef PARTICLE_MODEL
                        constexpr int particleInterval = static_cast<int>(PARTICLES_SAVE);
                        if constexpr (particleInterval > 0) particleSave = !(aux % particleInterval);
                    #endif //PARTICLE MODEL
                }
        #pragma warning(pop)
    }
    

} SaveField;
#endif //__SAVEFIELD_STRUCTS_H
