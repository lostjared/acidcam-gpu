#include <QApplication>
#include "midi_window.hpp"

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);
    app.setApplicationName("ACMX2 MIDI Map");
    app.setOrganizationName("LostSideDead");

    MidiMapWindow window;
    window.show();

    return app.exec();
}
