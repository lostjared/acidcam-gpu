#ifndef FIND_SHADER_HPP
#define FIND_SHADER_HPP

/**
 * @file find-shader.hpp
 * @brief Regular-expression search dialog for shader source libraries.
 */

#include <QDialog>
#include <QString>

class QCheckBox;
class QCloseEvent;
class QLabel;
class QLineEdit;
class QPushButton;
class QTreeWidget;
class QTreeWidgetItem;

/**
 * @brief Search all GLSL source files beneath an active shader library.
 */
class FindShaderDialog : public QDialog {
    Q_OBJECT

  public:
    explicit FindShaderDialog(const QString &shaderPath, QWidget *parent = nullptr);

  signals:
    /// @brief Emitted when the user opens one search result.
    void resultActivated(const QString &filePath, int lineNumber,
                         int columnNumber, int matchLength);

  protected:
    void closeEvent(QCloseEvent *event) override;

  private:
    void performSearch();
    void openResult(QTreeWidgetItem *item);
    void updateOpenButton();

    QString shaderPath;
    QLineEdit *patternEdit = nullptr;
    QCheckBox *caseSensitiveCheck = nullptr;
    QTreeWidget *resultsTree = nullptr;
    QLabel *statusLabel = nullptr;
    QPushButton *openButton = nullptr;
};

#endif
