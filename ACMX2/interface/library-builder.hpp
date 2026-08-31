#ifndef LIBRARY_BUILDER_HPP
#define LIBRARY_BUILDER_HPP

/**
 * @file library-builder.hpp
 * @brief Dialog for assembling portable shader libraries.
 */

#include "backend.hpp"
#include <QDialog>
#include <QStringList>

class QCheckBox;
class QLabel;
class QListWidget;
class QPushButton;

/**
 * @brief Builds an ordered library from fragment and compute shader files.
 */
class LibraryBuilderDialog : public QDialog {
    Q_OBJECT

  public:
    explicit LibraryBuilderDialog(acmx2::Backend backend,
                                  QWidget *parent = nullptr);
    /// @brief Return the backend whose source format this dialog exports.
    acmx2::Backend selectedBackend() const;

  signals:
    /// @brief Emitted after a complete library has been exported successfully.
    void libraryExported(const QString &directory);

  private slots:
    void addFiles();
    void addFolder();
    void openLibrary();
    void removeSelected();
    void clearShaders();
    void exportLibrary();
    void updateControls();

  private:
    bool addShader(const QString &filePath, bool showErrors = true);
    int addShaderFiles(const QStringList &filePaths);
    QString shaderFilter() const;
    QStringList selectedSourcePaths() const;

    QListWidget *shaderList = nullptr;
    QLabel *summaryLabel = nullptr;
    QCheckBox *recursiveCheck = nullptr;
    QPushButton *removeButton = nullptr;
    QPushButton *exportButton = nullptr;
    bool exportInProgress = false;
    acmx2::Backend backend;
};

#endif
