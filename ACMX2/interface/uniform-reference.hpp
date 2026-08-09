#ifndef UNIFORM_REFERENCE_HPP
#define UNIFORM_REFERENCE_HPP

#include <QDialog>

class QLineEdit;
class QListWidget;
class QListWidgetItem;
class QPlainTextEdit;

/**
 * @brief Searchable reference for uniforms supplied by the ACMX2 runtime.
 */
class UniformReferenceDialog : public QDialog {
    Q_OBJECT

  public:
    explicit UniformReferenceDialog(QWidget *parent = nullptr);

  private slots:
    void filterUniforms(const QString &text);
    void showUniformDetails(QListWidgetItem *current);

  private:
    void populateUniforms();

    QLineEdit *searchEdit = nullptr;
    QListWidget *uniformList = nullptr;
    QPlainTextEdit *descriptionView = nullptr;
};

#endif
