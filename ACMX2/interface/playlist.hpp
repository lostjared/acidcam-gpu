#ifndef __PLAYLIST_HPP__
#define __PLAYLIST_HPP__

/**
 * @file playlist.hpp
 * @brief Dialog for building nested shader playlists.
 */

#include <QCheckBox>
#include <QComboBox>
#include <QDialog>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QSortFilterProxyModel>
#include <QSpinBox>
#include <QStandardItemModel>
#include <QStringList>
#include <QTreeWidget>
#include <QVBoxLayout>

/**
 * @brief Playlist editor dialog with tree nodes and ordered shader entries.
 */
class PlaylistDialog : public QDialog {
    Q_OBJECT
  public:
    explicit PlaylistDialog(const QStringList &shaderNames, QWidget *parent = nullptr);

    /// @brief Return whether playlist mode is enabled.
    bool isPlaylistEnabled() const;
    /// @brief Return flattened selection of shaders in current playlist.
    QStringList getSelectedShaderNames() const;
    /// @brief Return tree representation as node-name and shader-list pairs.
    QList<QPair<QString, QStringList>> getPlaylistTree() const;
    /// @brief Return current playlist file path.
    QString getPlaylistFile() const;
    /// @brief Return frames-per-shader threshold for autopilot mode (0 = disabled).
    int getAutopilotFrames() const;

    void setEnabled(bool enabled);
    void setSelectedShaderNames(const QStringList &names);
    void setPlaylistTree(const QList<QPair<QString, QStringList>> &tree);
    void setPlaylistFile(const QString &path);
    /// @brief Set frames-per-shader threshold for autopilot mode.
    void setAutopilotFrames(int frames);
    void updateShaderList(const QStringList &shaderNames);

  public slots:
    void addNode();
    void renameNode();
    void removeNode();
    void addShader();
    void removeShader();
    void moveUp();
    void moveDown();
    void clearAll();
    void filterSearchChanged(const QString &text);
    void savePlaylist();
    void loadPlaylist();

  private:
    void setupUI();
    void loadShaders(const QStringList &shaderNames);
    QTreeWidgetItem *currentNodeItem() const;

    QCheckBox *enableCheckBox;
    QComboBox *shaderComboBox;
    QLineEdit *searchLineEdit;
    QTreeWidget *playlistTree;
    QPushButton *addNodeButton;
    QPushButton *renameNodeButton;
    QPushButton *removeNodeButton;
    QPushButton *addButton;
    QPushButton *removeButton;
    QPushButton *upButton;
    QPushButton *downButton;
    QPushButton *clearButton;
    QPushButton *saveButton;
    QPushButton *loadButton;
    QPushButton *okButton;
    QPushButton *cancelButton;

    QSpinBox *autopilotFramesSpinBox = nullptr;

    QStandardItemModel *shaderModel;
    QSortFilterProxyModel *proxyModel;

    QMap<QString, int> shaderNameToIndex;
    QStringList shaderNamesList;
    QString playlistFilePath;
};

#endif
