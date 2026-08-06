#ifndef AUDIO_PLAYLIST_HPP
#define AUDIO_PLAYLIST_HPP

#include <QDialog>
#include <QString>

class QLabel;
class QListWidget;
class QPushButton;

/**
 * @brief Editor for creating and modifying ordered M3U audio playlists.
 */
class AudioPlaylistDialog : public QDialog {
    Q_OBJECT

  public:
    explicit AudioPlaylistDialog(const QString &playlistPath = QString(),
                                 QWidget *parent = nullptr);

    /// @brief Return the playlist most recently opened or saved.
    QString playlistPath() const;

  private slots:
    void addTracks();
    void removeSelected();
    void moveUp();
    void moveDown();
    void sortTracks();
    void shuffleTracks();
    void clearTracks();
    void openPlaylist();
    void savePlaylist();
    void savePlaylistAs();
    void finishEditing();

  protected:
    void reject() override;

  private:
    void setupUi();
    bool loadPlaylist(const QString &path);
    bool writePlaylist(const QString &path);
    bool confirmDiscardChanges();
    void updateWindowState();
    QString itemPath(int row) const;
    void setDirty(bool dirty = true);

    QListWidget *trackList = nullptr;
    QLabel *pathLabel = nullptr;
    QPushButton *removeButton = nullptr;
    QPushButton *upButton = nullptr;
    QPushButton *downButton = nullptr;
    QPushButton *sortButton = nullptr;
    QPushButton *shuffleButton = nullptr;
    QPushButton *clearButton = nullptr;
    QPushButton *saveButton = nullptr;
    QString currentPlaylistPath;
    bool dirty = false;
};

#endif
