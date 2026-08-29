#include "midi_window.hpp"
#include <QApplication>

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);
    app.setApplicationName("ACMX MIDI Map");
    app.setOrganizationName("LostSideDead");

    MidiMapWindow window;
    window.show();

    return app.exec();
}
