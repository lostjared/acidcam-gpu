#ifndef UNIFORM_REFERENCE_HPP
#define UNIFORM_REFERENCE_HPP

#include "backend.hpp"

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
    explicit UniformReferenceDialog(acmx2::Backend backend,
                                    QWidget *parent = nullptr);
    void setBackend(acmx2::Backend backend);

  private slots:
    void filterUniforms(const QString &text);
    void showUniformDetails(QListWidgetItem *current);

  private:
    void populateUniforms();

    acmx2::Backend activeBackend = acmx2::Backend::Acmx2;
    QLineEdit *searchEdit = nullptr;
    QListWidget *uniformList = nullptr;
    QPlainTextEdit *descriptionView = nullptr;
};

#endif
