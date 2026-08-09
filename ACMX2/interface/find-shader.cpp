#include "find-shader.hpp"

#include <QApplication>
#include <QCheckBox>
#include <QCloseEvent>
#include <QDialogButtonBox>
#include <QDir>
#include <QDirIterator>
#include <QFile>
#include <QFileInfo>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QRegularExpression>
#include <QSettings>
#include <QTextStream>
#include <QTreeWidget>
#include <QVBoxLayout>

namespace {
    constexpr int FILE_PATH_ROLE = Qt::UserRole;
    constexpr int LINE_NUMBER_ROLE = Qt::UserRole + 1;
    constexpr int COLUMN_NUMBER_ROLE = Qt::UserRole + 2;
    constexpr int MATCH_LENGTH_ROLE = Qt::UserRole + 3;
    constexpr int MAX_RESULTS = 10000;
} // namespace

FindShaderDialog::FindShaderDialog(const QString &shaderPath, QWidget *parent)
    : QDialog(parent), shaderPath(QFileInfo(shaderPath).absoluteFilePath()) {
    setWindowTitle(tr("Find in Shader Files"));
    setAttribute(Qt::WA_DeleteOnClose);
    setModal(false);

    auto *layout = new QVBoxLayout(this);
    layout->addWidget(new QLabel(
        tr("Search the active shader library using a regular expression:"), this));
    auto *pathLabel = new QLabel(tr("Library: %1").arg(this->shaderPath), this);
    pathLabel->setTextInteractionFlags(Qt::TextSelectableByMouse);
    layout->addWidget(pathLabel);

    auto *searchRow = new QHBoxLayout();
    patternEdit = new QLineEdit(this);
    patternEdit->setPlaceholderText(tr("Regular expression, for example: uniform\\s+float"));
    patternEdit->setClearButtonEnabled(true);
    auto *searchButton = new QPushButton(tr("Search"), this);
    searchButton->setDefault(true);
    searchRow->addWidget(patternEdit, 1);
    searchRow->addWidget(searchButton);
    layout->addLayout(searchRow);

    caseSensitiveCheck = new QCheckBox(tr("Case sensitive"), this);
    layout->addWidget(caseSensitiveCheck);

    resultsTree = new QTreeWidget(this);
    resultsTree->setColumnCount(4);
    resultsTree->setHeaderLabels(
        {tr("Shader"), tr("Line"), tr("Match"), tr("Source")});
    resultsTree->setRootIsDecorated(false);
    resultsTree->setAlternatingRowColors(false);
    resultsTree->setSelectionBehavior(QAbstractItemView::SelectRows);
    resultsTree->setSelectionMode(QAbstractItemView::SingleSelection);
    resultsTree->setSortingEnabled(true);
    resultsTree->header()->setSectionResizeMode(0, QHeaderView::ResizeToContents);
    resultsTree->header()->setSectionResizeMode(1, QHeaderView::ResizeToContents);
    resultsTree->header()->setSectionResizeMode(2, QHeaderView::ResizeToContents);
    resultsTree->header()->setSectionResizeMode(3, QHeaderView::Stretch);
    layout->addWidget(resultsTree, 1);

    statusLabel = new QLabel(tr("Enter a regular expression to search."), this);
    layout->addWidget(statusLabel);

    auto *buttonBox = new QDialogButtonBox(QDialogButtonBox::Close, this);
    openButton = buttonBox->addButton(tr("Open Result"), QDialogButtonBox::ActionRole);
    openButton->setEnabled(false);
    layout->addWidget(buttonBox);

    connect(searchButton, &QPushButton::clicked, this, &FindShaderDialog::performSearch);
    connect(patternEdit, &QLineEdit::returnPressed, this, &FindShaderDialog::performSearch);
    connect(patternEdit, &QLineEdit::textChanged, this, [this]() {
        statusLabel->clear();
    });
    connect(resultsTree, &QTreeWidget::itemSelectionChanged,
            this, &FindShaderDialog::updateOpenButton);
    connect(resultsTree, &QTreeWidget::itemActivated,
            this, [this](QTreeWidgetItem *item, int) { openResult(item); });
    connect(openButton, &QPushButton::clicked, this, [this]() {
        openResult(resultsTree->currentItem());
    });
    connect(buttonBox, &QDialogButtonBox::rejected, this, &QDialog::close);

    QSettings settings("LostSideDead");
    caseSensitiveCheck->setChecked(
        settings.value("findInFiles/caseSensitive", false).toBool());
    if (!restoreGeometry(settings.value("findInFiles/geometry").toByteArray()))
        resize(950, 600);
    patternEdit->setFocus();
}

void FindShaderDialog::closeEvent(QCloseEvent *event) {
    QSettings("LostSideDead").setValue("findInFiles/geometry", saveGeometry());
    QDialog::closeEvent(event);
}

void FindShaderDialog::performSearch() {
    resultsTree->clear();
    updateOpenButton();

    const QString pattern = patternEdit->text();
    if (pattern.isEmpty()) {
        statusLabel->setText(tr("Enter a regular expression to search."));
        return;
    }

    QRegularExpression::PatternOptions options;
    if (!caseSensitiveCheck->isChecked())
        options |= QRegularExpression::CaseInsensitiveOption;
    const QRegularExpression expression(pattern, options);
    if (!expression.isValid()) {
        statusLabel->setText(
            tr("Invalid regular expression at position %1: %2")
                .arg(expression.patternErrorOffset())
                .arg(expression.errorString()));
        return;
    }

    QSettings settings("LostSideDead");
    settings.setValue("findInFiles/caseSensitive", caseSensitiveCheck->isChecked());
    settings.setValue("findInFiles/geometry", saveGeometry());

    QApplication::setOverrideCursor(Qt::WaitCursor);
    resultsTree->setSortingEnabled(false);

    const QStringList filters = {"*.glsl", "*.frag", "*.vert", "*.comp"};
    QDirIterator files(shaderPath, filters, QDir::Files | QDir::Readable,
                       QDirIterator::Subdirectories);
    int fileCount = 0;
    int resultCount = 0;
    bool limitReached = false;
    const QDir root(shaderPath);

    while (files.hasNext() && !limitReached) {
        const QString filePath = files.next();
        QFile file(filePath);
        if (!file.open(QIODevice::ReadOnly | QIODevice::Text))
            continue;
        ++fileCount;

        QTextStream input(&file);
        int lineNumber = 0;
        while (!input.atEnd() && !limitReached) {
            const QString line = input.readLine();
            ++lineNumber;
            QRegularExpressionMatchIterator matches = expression.globalMatch(line);
            while (matches.hasNext()) {
                const QRegularExpressionMatch match = matches.next();
                auto *item = new QTreeWidgetItem(resultsTree);
                item->setText(0, root.relativeFilePath(filePath));
                item->setText(1, QString::number(lineNumber));
                item->setText(2, match.captured(0).isEmpty()
                                     ? tr("(zero-length match)")
                                     : match.captured(0));
                item->setText(3, line.trimmed());
                item->setData(0, FILE_PATH_ROLE, QFileInfo(filePath).absoluteFilePath());
                item->setData(0, LINE_NUMBER_ROLE, lineNumber);
                item->setData(0, COLUMN_NUMBER_ROLE, match.capturedStart());
                item->setData(0, MATCH_LENGTH_ROLE, match.capturedLength());
                ++resultCount;
                if (resultCount >= MAX_RESULTS) {
                    limitReached = true;
                    break;
                }
            }
        }
    }

    resultsTree->setSortingEnabled(true);
    resultsTree->sortItems(0, Qt::AscendingOrder);
    QApplication::restoreOverrideCursor();

    if (limitReached) {
        statusLabel->setText(
            tr("Showing the first %1 matches from %2 shader files.")
                .arg(resultCount)
                .arg(fileCount));
    } else {
        statusLabel->setText(
            tr("Found %1 match(es) in %2 shader file(s).")
                .arg(resultCount)
                .arg(fileCount));
    }
}

void FindShaderDialog::openResult(QTreeWidgetItem *item) {
    if (!item)
        return;
    emit resultActivated(
        item->data(0, FILE_PATH_ROLE).toString(),
        item->data(0, LINE_NUMBER_ROLE).toInt(),
        item->data(0, COLUMN_NUMBER_ROLE).toInt(),
        item->data(0, MATCH_LENGTH_ROLE).toInt());
}

void FindShaderDialog::updateOpenButton() {
    openButton->setEnabled(resultsTree->currentItem() != nullptr);
}
