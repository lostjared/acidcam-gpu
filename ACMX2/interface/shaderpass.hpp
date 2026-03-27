#ifndef __SHADERPASS_HPP__
#define __SHADERPASS_HPP__

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

class ShaderPassDialog : public QDialog {
    Q_OBJECT
  public:
    explicit ShaderPassDialog(const QStringList &shaderNames, QWidget *parent = nullptr);

    bool isShaderPassEnabled() const;
    QStringList getSelectedShaderIndices() const;
    QString getShaderPassArgument() const;
    QStringList getSelectedShaderNames() const;

    // Set the current state
    void setEnabled(bool enabled);
    void setSelectedIndices(const QStringList &indices);
    void setSelectedShaderNames(const QStringList &names);
    void updateShaderList(const QStringList &shaderNames);

  public slots:
    void addShader();
    void removeShader();
    void moveUp();
    void moveDown();
    void clearAll();
    void filterSearchChanged(const QString &text);

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
    QPushButton *okButton;
    QPushButton *cancelButton;

    QStandardItemModel *shaderModel;
    QSortFilterProxyModel *proxyModel;

    QMap<QString, int> shaderNameToIndex;
    QStringList shaderNamesList;
};

#endif
