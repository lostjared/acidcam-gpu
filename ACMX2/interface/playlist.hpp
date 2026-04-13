#ifndef __PLAYLIST_HPP__
#define __PLAYLIST_HPP__

#include <QCheckBox>
#include <QComboBox>
#include <QDialog>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QListWidget>
#include <QPushButton>
#include <QSortFilterProxyModel>
#include <QStandardItemModel>
#include <QStringList>
#include <QVBoxLayout>

class PlaylistDialog : public QDialog {
    Q_OBJECT
  public:
    explicit PlaylistDialog(const QStringList &shaderNames, QWidget *parent = nullptr);

    bool isPlaylistEnabled() const;
    QStringList getSelectedShaderNames() const;
    QString getPlaylistFile() const;

    void setEnabled(bool enabled);
    void setSelectedShaderNames(const QStringList &names);
    void setPlaylistFile(const QString &path);
    void updateShaderList(const QStringList &shaderNames);

  public slots:
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

    QCheckBox *enableCheckBox;
    QComboBox *shaderComboBox;
    QLineEdit *searchLineEdit;
    QListWidget *selectedShadersList;
    QPushButton *addButton;
    QPushButton *removeButton;
    QPushButton *upButton;
    QPushButton *downButton;
    QPushButton *clearButton;
    QPushButton *saveButton;
    QPushButton *loadButton;
    QPushButton *okButton;
    QPushButton *cancelButton;

    QStandardItemModel *shaderModel;
    QSortFilterProxyModel *proxyModel;

    QMap<QString, int> shaderNameToIndex;
    QStringList shaderNamesList;
    QString playlistFilePath;
};

#endif
