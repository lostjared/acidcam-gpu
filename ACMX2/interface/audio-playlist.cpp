#include "audio-playlist.hpp"
#include "custom_style.hpp"

#include <QAbstractItemModel>
#include <QDir>
#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QHBoxLayout>
#include <QLabel>
#include <QListWidget>
#include <QMessageBox>
#include <QPushButton>
#include <QRandomGenerator>
#include <QSaveFile>
#include <QSettings>
#include <QVBoxLayout>

#include <algorithm>
#include <functional>
#include <random>
#include <vector>

namespace {

    constexpr int TRACK_PATH_ROLE = Qt::UserRole;

    bool isUrl(const QString &path) {
        return path.contains("://");
    }

    QString displayName(const QString &path) {
        if (isUrl(path))
            return path;
        const QString name = QFileInfo(path).fileName();
        return name.isEmpty() ? path : name;
    }

} // namespace

AudioPlaylistDialog::AudioPlaylistDialog(const QString &playlistPath,
                                         QWidget *parent)
    : QDialog(parent) {
    setupUi();
    if (!playlistPath.isEmpty() && QFileInfo::exists(playlistPath))
        loadPlaylist(playlistPath);
    updateWindowState();
}

QString AudioPlaylistDialog::playlistPath() const {
    return currentPlaylistPath;
}

void AudioPlaylistDialog::setupUi() {
    setWindowTitle("Audio M3U Playlist Editor");
    setMinimumSize(720, 520);

    QVBoxLayout *mainLayout = new QVBoxLayout(this);
    QLabel *instructions = new QLabel(
        "Add audio tracks in playback order. You can reorder, sort, shuffle, "
        "open an existing M3U playlist, or save the current list as M3U.",
        this);
    instructions->setWordWrap(true);
    mainLayout->addWidget(instructions);

    pathLabel = new QLabel(this);
    pathLabel->setTextInteractionFlags(Qt::TextSelectableByMouse);
    pathLabel->setWordWrap(true);
    mainLayout->addWidget(pathLabel);

    trackList = new QListWidget(this);
    trackList->setSelectionMode(QAbstractItemView::ExtendedSelection);
    trackList->setDragDropMode(QAbstractItemView::InternalMove);
    trackList->setDefaultDropAction(Qt::MoveAction);
    mainLayout->addWidget(trackList, 1);

    QHBoxLayout *trackButtons = new QHBoxLayout();
    QPushButton *addButton = new QPushButton("Add Tracks...", this);
    removeButton = new QPushButton("Remove", this);
    upButton = new QPushButton("Move Up", this);
    downButton = new QPushButton("Move Down", this);
    sortButton = new QPushButton("Sort", this);
    shuffleButton = new QPushButton("Shuffle", this);
    clearButton = new QPushButton("Clear", this);
    trackButtons->addWidget(addButton);
    trackButtons->addWidget(removeButton);
    trackButtons->addWidget(upButton);
    trackButtons->addWidget(downButton);
    trackButtons->addWidget(sortButton);
    trackButtons->addWidget(shuffleButton);
    trackButtons->addWidget(clearButton);
    mainLayout->addLayout(trackButtons);

    QHBoxLayout *fileButtons = new QHBoxLayout();
    QPushButton *openButton = new QPushButton("Open M3U...", this);
    saveButton = new QPushButton("Save", this);
    QPushButton *saveAsButton = new QPushButton("Save As...", this);
    QPushButton *doneButton = new QPushButton("Done", this);
    QPushButton *cancelButton = new QPushButton("Cancel", this);
    fileButtons->addWidget(openButton);
    fileButtons->addWidget(saveButton);
    fileButtons->addWidget(saveAsButton);
    fileButtons->addStretch();
    fileButtons->addWidget(doneButton);
    fileButtons->addWidget(cancelButton);
    mainLayout->addLayout(fileButtons);

    connect(addButton, &QPushButton::clicked, this,
            &AudioPlaylistDialog::addTracks);
    connect(removeButton, &QPushButton::clicked, this,
            &AudioPlaylistDialog::removeSelected);
    connect(upButton, &QPushButton::clicked, this,
            &AudioPlaylistDialog::moveUp);
    connect(downButton, &QPushButton::clicked, this,
            &AudioPlaylistDialog::moveDown);
    connect(sortButton, &QPushButton::clicked, this,
            &AudioPlaylistDialog::sortTracks);
    connect(shuffleButton, &QPushButton::clicked, this,
            &AudioPlaylistDialog::shuffleTracks);
    connect(clearButton, &QPushButton::clicked, this,
            &AudioPlaylistDialog::clearTracks);
    connect(openButton, &QPushButton::clicked, this,
            &AudioPlaylistDialog::openPlaylist);
    connect(saveButton, &QPushButton::clicked, this,
            &AudioPlaylistDialog::savePlaylist);
    connect(saveAsButton, &QPushButton::clicked, this,
            &AudioPlaylistDialog::savePlaylistAs);
    connect(doneButton, &QPushButton::clicked, this,
            &AudioPlaylistDialog::finishEditing);
    connect(cancelButton, &QPushButton::clicked, this,
            &AudioPlaylistDialog::reject);
    connect(trackList, &QListWidget::itemSelectionChanged, this,
            &AudioPlaylistDialog::updateWindowState);
    connect(trackList->model(), &QAbstractItemModel::rowsMoved, this,
            [this]() { setDirty(); });

    acmx2::applyCustomStyleIfEnabled(this);
}

void AudioPlaylistDialog::addTracks() {
    QSettings settings("LostSideDead", "acmx2");
    const QString startDirectory =
        settings.value("audio/playlist_track_dir", QString()).toString();
    const QStringList paths = QFileDialog::getOpenFileNames(
        this, "Add Audio Tracks", startDirectory,
        "Audio Files (*.wav *.mp3 *.flac *.aac *.ogg *.m4a *.wma *.opus "
        "*.mp4 *.mkv *.mov *.avi);;All Files (*)");
    if (paths.isEmpty())
        return;

    settings.setValue("audio/playlist_track_dir",
                      QFileInfo(paths.front()).absolutePath());
    for (const QString &path : paths) {
        const QString absolutePath = QFileInfo(path).absoluteFilePath();
        auto *item = new QListWidgetItem(displayName(absolutePath), trackList);
        item->setData(TRACK_PATH_ROLE, absolutePath);
        item->setToolTip(absolutePath);
    }
    setDirty();
}

void AudioPlaylistDialog::removeSelected() {
    QList<int> rows;
    for (QListWidgetItem *item : trackList->selectedItems())
        rows.push_back(trackList->row(item));
    std::sort(rows.begin(), rows.end(), std::greater<int>());
    for (int row : rows)
        delete trackList->takeItem(row);
    if (!rows.isEmpty())
        setDirty();
}

void AudioPlaylistDialog::moveUp() {
    QList<int> rows;
    for (QListWidgetItem *item : trackList->selectedItems())
        rows.push_back(trackList->row(item));
    std::sort(rows.begin(), rows.end());
    if (rows.isEmpty() || rows.front() == 0)
        return;
    for (int row : rows) {
        QListWidgetItem *item = trackList->takeItem(row);
        trackList->insertItem(row - 1, item);
        item->setSelected(true);
    }
    trackList->setCurrentRow(rows.front() - 1);
    setDirty();
}

void AudioPlaylistDialog::moveDown() {
    QList<int> rows;
    for (QListWidgetItem *item : trackList->selectedItems())
        rows.push_back(trackList->row(item));
    std::sort(rows.begin(), rows.end(), std::greater<int>());
    if (rows.isEmpty() || rows.front() == trackList->count() - 1)
        return;
    for (int row : rows) {
        QListWidgetItem *item = trackList->takeItem(row);
        trackList->insertItem(row + 1, item);
        item->setSelected(true);
    }
    trackList->setCurrentRow(rows.back() + 1);
    setDirty();
}

void AudioPlaylistDialog::sortTracks() {
    std::vector<QString> paths;
    paths.reserve(static_cast<std::size_t>(trackList->count()));
    for (int row = 0; row < trackList->count(); ++row)
        paths.push_back(itemPath(row));
    std::stable_sort(paths.begin(), paths.end(),
                     [](const QString &left, const QString &right) {
                         return QString::compare(displayName(left),
                                                 displayName(right),
                                                 Qt::CaseInsensitive) < 0;
                     });

    trackList->clear();
    for (const QString &path : paths) {
        auto *item = new QListWidgetItem(displayName(path), trackList);
        item->setData(TRACK_PATH_ROLE, path);
        item->setToolTip(path);
    }
    if (!paths.empty())
        setDirty();
}

void AudioPlaylistDialog::shuffleTracks() {
    std::vector<QString> paths;
    paths.reserve(static_cast<std::size_t>(trackList->count()));
    for (int row = 0; row < trackList->count(); ++row)
        paths.push_back(itemPath(row));
    std::mt19937 generator(QRandomGenerator::global()->generate());
    std::shuffle(paths.begin(), paths.end(), generator);

    trackList->clear();
    for (const QString &path : paths) {
        auto *item = new QListWidgetItem(displayName(path), trackList);
        item->setData(TRACK_PATH_ROLE, path);
        item->setToolTip(path);
    }
    if (paths.size() > 1)
        setDirty();
}

void AudioPlaylistDialog::clearTracks() {
    if (trackList->count() == 0)
        return;
    if (QMessageBox::question(this, "Clear Playlist",
                              "Remove every track from the playlist?") !=
        QMessageBox::Yes)
        return;
    trackList->clear();
    setDirty();
}

void AudioPlaylistDialog::openPlaylist() {
    QSettings settings("LostSideDead", "acmx2");
    const QString startDirectory =
        settings.value("audio/playlist_dir", QString()).toString();
    const QString path = QFileDialog::getOpenFileName(
        this, "Open Audio Playlist", startDirectory,
        "M3U Playlists (*.m3u *.m3u8)");
    if (!path.isEmpty() && confirmDiscardChanges())
        loadPlaylist(path);
}

bool AudioPlaylistDialog::loadPlaylist(const QString &path) {
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        QMessageBox::critical(this, "Open Playlist",
                              "Could not open the playlist:\n" + path +
                                  "\n\n" + file.errorString());
        return false;
    }

    const QDir playlistDirectory = QFileInfo(path).absoluteDir();
    const QList<QByteArray> lines = file.readAll().split('\n');
    trackList->clear();
    for (QByteArray encodedLine : lines) {
        if (encodedLine.endsWith('\r'))
            encodedLine.chop(1);
        if (encodedLine.startsWith("\xef\xbb\xbf"))
            encodedLine.remove(0, 3);
        QString entry = QString::fromUtf8(encodedLine).trimmed();
        if (entry.isEmpty() || entry.startsWith('#'))
            continue;

        QString resolvedPath = entry;
        if (!isUrl(entry) && QFileInfo(entry).isRelative())
            resolvedPath = playlistDirectory.absoluteFilePath(entry);
        if (!isUrl(resolvedPath))
            resolvedPath = QDir::cleanPath(resolvedPath);
        auto *item = new QListWidgetItem(displayName(resolvedPath), trackList);
        item->setData(TRACK_PATH_ROLE, resolvedPath);
        item->setToolTip(resolvedPath);
    }

    currentPlaylistPath = QFileInfo(path).absoluteFilePath();
    QSettings settings("LostSideDead", "acmx2");
    settings.setValue("audio/playlist_dir",
                      QFileInfo(currentPlaylistPath).absolutePath());
    setDirty(false);
    return true;
}

void AudioPlaylistDialog::savePlaylist() {
    if (currentPlaylistPath.isEmpty()) {
        savePlaylistAs();
        return;
    }
    writePlaylist(currentPlaylistPath);
}

void AudioPlaylistDialog::savePlaylistAs() {
    QSettings settings("LostSideDead", "acmx2");
    QString suggestedPath = currentPlaylistPath;
    if (suggestedPath.isEmpty()) {
        const QString directory =
            settings.value("audio/playlist_dir", QString()).toString();
        suggestedPath = QDir(directory).filePath("audio-playlist.m3u");
    }
    QString path = QFileDialog::getSaveFileName(
        this, "Save Audio Playlist", suggestedPath,
        "M3U Playlists (*.m3u);;UTF-8 M3U Playlists (*.m3u8)");
    if (path.isEmpty())
        return;
    if (!path.endsWith(".m3u", Qt::CaseInsensitive) &&
        !path.endsWith(".m3u8", Qt::CaseInsensitive))
        path += ".m3u";
    writePlaylist(path);
}

bool AudioPlaylistDialog::writePlaylist(const QString &path) {
    if (trackList->count() == 0) {
        QMessageBox::information(this, "Save Playlist",
                                 "Add at least one audio track before saving.");
        return false;
    }

    const QFileInfo playlistInfo(path);
    const QDir playlistDirectory = playlistInfo.absoluteDir();
    QByteArray contents("#EXTM3U\n");
    for (int row = 0; row < trackList->count(); ++row) {
        QString entry = itemPath(row);
        if (!isUrl(entry))
            entry = playlistDirectory.relativeFilePath(entry);
        contents += entry.toUtf8();
        contents += '\n';
    }

    QSaveFile file(path);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text) ||
        file.write(contents) != contents.size() || !file.commit()) {
        QMessageBox::critical(this, "Save Playlist",
                              "Could not save the playlist:\n" + path +
                                  "\n\n" + file.errorString());
        return false;
    }

    currentPlaylistPath = playlistInfo.absoluteFilePath();
    QSettings settings("LostSideDead", "acmx2");
    settings.setValue("audio/playlist_dir", playlistInfo.absolutePath());
    setDirty(false);
    return true;
}

void AudioPlaylistDialog::finishEditing() {
    if (dirty) {
        const QMessageBox::StandardButton choice = QMessageBox::question(
            this, "Save Playlist", "Save the playlist before closing?",
            QMessageBox::Save | QMessageBox::Discard | QMessageBox::Cancel,
            QMessageBox::Save);
        if (choice == QMessageBox::Cancel)
            return;
        if (choice == QMessageBox::Save) {
            if (currentPlaylistPath.isEmpty())
                savePlaylistAs();
            else
                writePlaylist(currentPlaylistPath);
            if (dirty)
                return;
        }
    }
    accept();
}

void AudioPlaylistDialog::reject() {
    if (confirmDiscardChanges())
        QDialog::reject();
}

bool AudioPlaylistDialog::confirmDiscardChanges() {
    if (!dirty)
        return true;
    const QMessageBox::StandardButton choice = QMessageBox::question(
        this, "Unsaved Playlist", "Discard the unsaved playlist changes?",
        QMessageBox::Discard | QMessageBox::Cancel, QMessageBox::Cancel);
    return choice == QMessageBox::Discard;
}

void AudioPlaylistDialog::updateWindowState() {
    const bool hasTracks = trackList->count() > 0;
    const bool hasSelection = !trackList->selectedItems().isEmpty();
    pathLabel->setText(currentPlaylistPath.isEmpty()
                           ? "Playlist: New playlist"
                           : "Playlist: " + currentPlaylistPath);
    removeButton->setEnabled(hasSelection);
    upButton->setEnabled(hasSelection);
    downButton->setEnabled(hasSelection);
    sortButton->setEnabled(hasTracks);
    shuffleButton->setEnabled(trackList->count() > 1);
    clearButton->setEnabled(hasTracks);
    saveButton->setEnabled(hasTracks);
    setWindowModified(dirty);
    setWindowTitle("Audio M3U Playlist Editor[*]");
}

QString AudioPlaylistDialog::itemPath(int row) const {
    QListWidgetItem *item = trackList->item(row);
    return item ? item->data(TRACK_PATH_ROLE).toString() : QString();
}

void AudioPlaylistDialog::setDirty(bool value) {
    dirty = value;
    updateWindowState();
}
