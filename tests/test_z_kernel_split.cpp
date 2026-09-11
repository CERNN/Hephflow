#include <iostream>
#include <vector>

static bool checkSplit(unsigned blocks) {
    std::vector<unsigned> visits(blocks, 0);
    auto launch = [&](unsigned first, unsigned count) {
        for (unsigned launchZ = 0; launchZ < count; ++launchZ)
            ++visits[first + launchZ];
    };

    if (blocks > 2) launch(1, blocks - 2);
    launch(0, 1);
    if (blocks > 1) launch(blocks - 1, 1);

    for (unsigned count : visits)
        if (count != 1) return false;
    return true;
}

int main() {
    const bool ok = checkSplit(1) && checkSplit(2) && checkSplit(3) && checkSplit(32);
    std::cout << "Z boundary/interior launch coverage " << (ok ? "PASS" : "FAIL") << '\n';
    return ok ? 0 : 1;
}
