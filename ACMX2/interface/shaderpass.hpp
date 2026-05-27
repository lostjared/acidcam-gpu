#ifndef __SHADERPASS_HPP__
#define __SHADERPASS_HPP__

/**
 * @file shaderpass.hpp
 * @brief Dialog for configuring ordered multi-pass shader execution.
 */

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

/**
 * @brief UI for selecting and ordering shader passes.
 */
class ShaderPassDialog : public QDialog {
    Q_OBJECT
  public:
    explicit ShaderPassDialog(const QStringList &shaderNames, QWidget *parent = nullptr);

    /// @brief Return whether shader-pass mode is enabled.
    bool isShaderPassEnabled() const;
    /// @brief Return selected pass indices corresponding to available shader list.
    QStringList getSelectedShaderIndices() const;
    /// @brief Build CLI argument payload for selected shader passes.
    QString getShaderPassArgument() const;
    /// @brief Return selected shader names in execution order.
    QStringList getSelectedShaderNames() const;

    // Set the current state
    void setEnabled(bool enabled);
    void setSelectedIndices(const QStringList &indices);
    void setSelectedShaderNames(const QStringList &names);
    void updateShaderList(const QStringList &shaderNames);

  signals:
    void settingsApplied(bool enabled, const QStringList &selectedShaderNames);

  public slots:
    void addShader();
    void removeShader();
    void moveUp();
    void moveDown();
    void clearAll();
    void filterSearchChanged(const QString &text);
    void saveShaderPass();
    void loadShaderPass();
    void applyChanges();

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
    QPushButton *applyButton;
    QPushButton *cancelButton;

    QStandardItemModel *shaderModel;
    QSortFilterProxyModel *proxyModel;

    QMap<QString, int> shaderNameToIndex;
    QStringList shaderNamesList;
};

#endif
