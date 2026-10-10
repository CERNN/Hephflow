#include "simulation.cuh"

#include <cstdlib>
#include <cstring>

int main(int argc, char** argv) {
    // Memory estimation is intentionally side-effect free: no CUDA context,
    // output folder, or simulation allocation is created on this path.
    if (argc == 2 && std::strcmp(argv[1], "--memory-estimate") == 0) {
        printCompileTimeMemoryEstimate();
        return 0;
    }

    folderSetup();

    SimulationContext context;
    if (!initializeSimulation(context)) {
        return EXIT_FAILURE;
    }

    for (; context.step < N_STEPS; ++context.step) {
        // SaveField translates compile-time output intervals into requests for
        // this step. Streaming parity selects the active AA/AB generation.
        SaveField saveField;
        saveField.flagsUpdate(context.step);
        context.stepParity = context.devices[0].effectiveStreamingParity(context.step);

        collideAndExchangeHalos(context, saveField);  // Main collide/stream phase.
        updateAuxiliaryPhysics(context);              // Derived phase/density fields.
        runHalfStepAndParticles(context);             // Post-collision coupled physics.
        processStepOutputs(context, saveField);       // Shared host snapshot consumers.
        runConfiguredTestMetric(context);             // Case-specific regression hook.
        processParticleOutputs(context, saveField);   // Particle trajectory output.
    }

    // Final output uses a fully synchronized field. Resource destruction is
    // kept separate so pending host-side writers are drained first.
    const dfloat mlups = finishSimulationTiming(context);
    writeFinalOutputs(context, mlups);
    destroySimulation(context);
    return 0;
}
