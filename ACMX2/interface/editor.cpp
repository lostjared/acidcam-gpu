#include "editor.hpp"
#include "custom_style.hpp"
#include <QAction>
#include <QApplication>
#include <QClipboard>
#include <QColor>
#include <QFile>
#include <QFileDialog>
#include <QFont>
#include <QInputDialog>
#include <QKeyEvent>
#include <QLabel>
#include <QMenuBar>
#include <QMessageBox>
#include <QMimeData>
#include <QPainter>
#include <QPlainTextEdit>
#include <QRegularExpression>
#include <QSaveFile>
#include <QSettings>
#include <QStatusBar>
#include <QSyntaxHighlighter>
#include <QTextBlock>
#include <QTextCharFormat>
#include <QTextCursor>
#include <QTextStream>
#include <QVBoxLayout>

// --- CustomTextEdit ---

CustomTextEdit::CustomTextEdit(QWidget *parent) : QPlainTextEdit(parent) {
    m_lineNumberArea = new LineNumberArea(this);

    connect(this, &QPlainTextEdit::blockCountChanged, this, &CustomTextEdit::updateLineNumberAreaWidth);
    connect(this, &QPlainTextEdit::updateRequest, this, &CustomTextEdit::updateLineNumberArea);
    connect(this, &QPlainTextEdit::cursorPositionChanged, this, &CustomTextEdit::highlightCurrentLine);
    connect(this, &QPlainTextEdit::cursorPositionChanged, this, &CustomTextEdit::matchBrackets);

    updateLineNumberAreaWidth(0);
    highlightCurrentLine();
}

int CustomTextEdit::lineNumberAreaWidth() {
    int digits = 1;
    int max = qMax(1, blockCount());
    while (max >= 10) {
        max /= 10;
        ++digits;
    }
    digits = qMax(digits, 3);
    return 10 + fontMetrics().horizontalAdvance(QLatin1Char('9')) * digits;
}

void CustomTextEdit::updateLineNumberAreaWidth(int /*newBlockCount*/) {
    setViewportMargins(lineNumberAreaWidth(), 0, 0, 0);
}

void CustomTextEdit::updateLineNumberArea(const QRect &rect, int dy) {
    if (dy)
        m_lineNumberArea->scroll(0, dy);
    else
        m_lineNumberArea->update(0, rect.y(), m_lineNumberArea->width(), rect.height());

    if (rect.contains(viewport()->rect()))
        updateLineNumberAreaWidth(0);
}

void CustomTextEdit::resizeEvent(QResizeEvent *event) {
    QPlainTextEdit::resizeEvent(event);
    QRect cr = contentsRect();
    m_lineNumberArea->setGeometry(QRect(cr.left(), cr.top(), lineNumberAreaWidth(), cr.height()));
}

void CustomTextEdit::lineNumberAreaPaintEvent(QPaintEvent *event) {
    QPainter painter(m_lineNumberArea);
    painter.fillRect(event->rect(), QColor(30, 30, 30));

    QTextBlock block = firstVisibleBlock();
    int blockNumber = block.blockNumber();
    int top = qRound(blockBoundingGeometry(block).translated(contentOffset()).top());
    int bottom = top + qRound(blockBoundingRect(block).height());

    while (block.isValid() && top <= event->rect().bottom()) {
        if (block.isVisible() && bottom >= event->rect().top()) {
            QString number = QString::number(blockNumber + 1);
            painter.setPen(QColor(120, 120, 120));
            if (blockNumber == textCursor().blockNumber())
                painter.setPen(QColor(220, 220, 220));
            painter.drawText(0, top, m_lineNumberArea->width() - 5, fontMetrics().height(),
                             Qt::AlignRight, number);
        }
        block = block.next();
        top = bottom;
        bottom = top + qRound(blockBoundingRect(block).height());
        ++blockNumber;
    }
}

void CustomTextEdit::highlightCurrentLine() {
    QList<QTextEdit::ExtraSelection> extraSelections;

    QTextEdit::ExtraSelection selection;
    selection.format.setBackground(QColor(40, 40, 50));
    selection.format.setProperty(QTextFormat::FullWidthSelection, true);
    selection.cursor = textCursor();
    selection.cursor.clearSelection();
    extraSelections.append(selection);

    setExtraSelections(extraSelections);
}

static QChar matchingBracket(QChar ch) {
    if (ch == '(')
        return ')';
    if (ch == ')')
        return '(';
    if (ch == '{')
        return '}';
    if (ch == '}')
        return '{';
    if (ch == '[')
        return ']';
    if (ch == ']')
        return '[';
    return QChar();
}

static bool isOpenBracket(QChar ch) {
    return ch == '(' || ch == '{' || ch == '[';
}

void CustomTextEdit::matchBrackets() {
    QList<QTextEdit::ExtraSelection> selections;

    // Keep current line highlight
    QTextEdit::ExtraSelection lineSelection;
    lineSelection.format.setBackground(QColor(40, 40, 50));
    lineSelection.format.setProperty(QTextFormat::FullWidthSelection, true);
    lineSelection.cursor = textCursor();
    lineSelection.cursor.clearSelection();
    selections.append(lineSelection);

    QTextCursor cursor = textCursor();
    QTextDocument *doc = document();
    int pos = cursor.position();

    auto tryMatch = [&](int checkPos) -> bool {
        if (checkPos < 0 || checkPos >= doc->characterCount())
            return false;

        QChar ch = doc->characterAt(checkPos);
        QChar match = matchingBracket(ch);
        if (match.isNull())
            return false;

        bool forward = isOpenBracket(ch);
        int depth = 1;
        int i = checkPos + (forward ? 1 : -1);

        while (i >= 0 && i < doc->characterCount() && depth > 0) {
            QChar c = doc->characterAt(i);
            if (c == ch)
                ++depth;
            else if (c == match)
                --depth;
            if (forward)
                ++i;
            else
                --i;
        }

        if (depth == 0) {
            int matchPos = forward ? i - 1 : i + 1;

            QTextEdit::ExtraSelection sel1;
            sel1.format.setBackground(QColor(80, 80, 120));
            sel1.cursor = QTextCursor(doc);
            sel1.cursor.setPosition(checkPos);
            sel1.cursor.movePosition(QTextCursor::Right, QTextCursor::KeepAnchor);
            selections.append(sel1);

            QTextEdit::ExtraSelection sel2;
            sel2.format.setBackground(QColor(80, 80, 120));
            sel2.cursor = QTextCursor(doc);
            sel2.cursor.setPosition(matchPos);
            sel2.cursor.movePosition(QTextCursor::Right, QTextCursor::KeepAnchor);
            selections.append(sel2);
            return true;
        }
        return false;
    };

    if (!tryMatch(pos))
        tryMatch(pos - 1);

    setExtraSelections(selections);
}

bool CustomTextEdit::hasMultiLineSelection() {
    QTextCursor cursor = textCursor();
    if (!cursor.hasSelection())
        return false;
    int startBlock = document()->findBlock(cursor.selectionStart()).blockNumber();
    int endBlock = document()->findBlock(cursor.selectionEnd()).blockNumber();
    return startBlock != endBlock;
}

void CustomTextEdit::indentSelection() {
    QTextCursor cursor = textCursor();
    int start = cursor.selectionStart();
    int end = cursor.selectionEnd();

    QTextBlock startBlock = document()->findBlock(start);
    QTextBlock endBlock = document()->findBlock(end);
    if (end > start && endBlock.position() == end)
        endBlock = endBlock.previous();

    cursor.beginEditBlock();
    QTextBlock block = startBlock;
    while (block.isValid() && block.blockNumber() <= endBlock.blockNumber()) {
        QTextCursor blockCursor(block);
        blockCursor.movePosition(QTextCursor::StartOfBlock);
        blockCursor.insertText("    ");
        block = block.next();
    }
    cursor.endEditBlock();
}

void CustomTextEdit::unindentSelection() {
    QTextCursor cursor = textCursor();
    int start = cursor.selectionStart();
    int end = cursor.selectionEnd();

    QTextBlock startBlock = document()->findBlock(start);
    QTextBlock endBlock = document()->findBlock(end);
    if (end > start && endBlock.position() == end)
        endBlock = endBlock.previous();

    cursor.beginEditBlock();
    QTextBlock block = startBlock;
    while (block.isValid() && block.blockNumber() <= endBlock.blockNumber()) {
        QString text = block.text();
        int spaces = 0;
        for (auto ch : text) {
            if (ch == ' ' && spaces < 4)
                ++spaces;
            else
                break;
        }
        if (spaces > 0) {
            QTextCursor blockCursor(block);
            blockCursor.movePosition(QTextCursor::StartOfBlock);
            blockCursor.movePosition(QTextCursor::Right, QTextCursor::KeepAnchor, spaces);
            blockCursor.removeSelectedText();
        }
        block = block.next();
    }
    cursor.endEditBlock();
}

void CustomTextEdit::autoIndentNewLine() {
    QTextCursor cursor = textCursor();
    QString blockText = cursor.block().text();
    int col = cursor.positionInBlock();

    QString beforeCursor = blockText.left(col);

    // Compute leading whitespace
    QString indent;
    for (auto ch : beforeCursor) {
        if (ch == ' ' || ch == '\t')
            indent += ch;
        else
            break;
    }

    QString trimmed = beforeCursor.trimmed();
    bool afterOpenBrace = trimmed.endsWith('{');

    QString afterCursor = blockText.mid(col);
    bool beforeCloseBrace = afterCursor.trimmed().startsWith('}');

    cursor.beginEditBlock();
    cursor.insertText("\n");

    if (afterOpenBrace && beforeCloseBrace) {
        // Between {} — create indented block
        cursor.insertText(indent + "    ");
        int cursorPos = cursor.position();
        cursor.insertText("\n" + indent);
        cursor.setPosition(cursorPos);
        setTextCursor(cursor);
    } else if (afterOpenBrace) {
        cursor.insertText(indent + "    ");
    } else {
        cursor.insertText(indent);
    }
    cursor.endEditBlock();

    if (!(afterOpenBrace && beforeCloseBrace))
        setTextCursor(cursor);

    ensureCursorVisible();
}

void CustomTextEdit::duplicateLine() {
    QTextCursor cursor = textCursor();
    cursor.beginEditBlock();
    cursor.movePosition(QTextCursor::StartOfBlock);
    cursor.movePosition(QTextCursor::EndOfBlock, QTextCursor::KeepAnchor);
    QString lineText = cursor.selectedText();
    cursor.movePosition(QTextCursor::EndOfBlock);
    cursor.insertText("\n" + lineText);
    cursor.endEditBlock();
    setTextCursor(cursor);
}

void CustomTextEdit::moveLineUp() {
    QTextCursor cursor = textCursor();
    if (cursor.blockNumber() == 0)
        return;

    cursor.beginEditBlock();
    cursor.movePosition(QTextCursor::StartOfBlock);
    cursor.movePosition(QTextCursor::EndOfBlock, QTextCursor::KeepAnchor);
    QString currentLine = cursor.selectedText();

    // Select the preceding newline + entire current line
    cursor.movePosition(QTextCursor::StartOfBlock);
    cursor.movePosition(QTextCursor::Left, QTextCursor::KeepAnchor);
    cursor.movePosition(QTextCursor::EndOfBlock, QTextCursor::KeepAnchor);
    cursor.removeSelectedText();

    // Insert above
    cursor.movePosition(QTextCursor::StartOfBlock);
    cursor.insertText(currentLine + "\n");
    cursor.movePosition(QTextCursor::Up);
    cursor.endEditBlock();
    setTextCursor(cursor);
}

void CustomTextEdit::moveLineDown() {
    QTextCursor cursor = textCursor();
    if (cursor.blockNumber() >= document()->blockCount() - 1)
        return;

    cursor.beginEditBlock();
    cursor.movePosition(QTextCursor::StartOfBlock);
    cursor.movePosition(QTextCursor::EndOfBlock, QTextCursor::KeepAnchor);
    QString currentLine = cursor.selectedText();

    // Select current line + trailing newline
    cursor.movePosition(QTextCursor::StartOfBlock);
    cursor.movePosition(QTextCursor::EndOfBlock, QTextCursor::KeepAnchor);
    cursor.movePosition(QTextCursor::Right, QTextCursor::KeepAnchor);
    cursor.removeSelectedText();

    // Insert below current position
    cursor.movePosition(QTextCursor::EndOfBlock);
    cursor.insertText("\n" + currentLine);
    cursor.endEditBlock();
    setTextCursor(cursor);
}

void CustomTextEdit::toggleComment() {
    QTextCursor cursor = textCursor();
    int start = cursor.selectionStart();
    int end = cursor.selectionEnd();

    QTextBlock startBlock = document()->findBlock(start);
    QTextBlock endBlock = document()->findBlock(end);
    if (!cursor.hasSelection())
        endBlock = startBlock;
    else if (endBlock.position() == end && end > start)
        endBlock = endBlock.previous();

    // Check if all selected lines are already commented
    bool allCommented = true;
    QTextBlock block = startBlock;
    while (block.isValid() && block.blockNumber() <= endBlock.blockNumber()) {
        QString trimmed = block.text().trimmed();
        if (!trimmed.isEmpty() && !trimmed.startsWith("//")) {
            allCommented = false;
            break;
        }
        block = block.next();
    }

    cursor.beginEditBlock();
    block = startBlock;
    while (block.isValid() && block.blockNumber() <= endBlock.blockNumber()) {
        QTextCursor blockCursor(block);
        if (allCommented) {
            QString text = block.text();
            int idx = text.indexOf("//");
            if (idx >= 0) {
                blockCursor.movePosition(QTextCursor::StartOfBlock);
                blockCursor.movePosition(QTextCursor::Right, QTextCursor::MoveAnchor, idx);
                int removeLen = 2;
                if (idx + 2 < text.length() && text[idx + 2] == ' ')
                    removeLen = 3;
                blockCursor.movePosition(QTextCursor::Right, QTextCursor::KeepAnchor, removeLen);
                blockCursor.removeSelectedText();
            }
        } else {
            blockCursor.movePosition(QTextCursor::StartOfBlock);
            blockCursor.insertText("// ");
        }
        block = block.next();
    }
    cursor.endEditBlock();
}

void CustomTextEdit::smartHome(bool shift) {
    QTextCursor cursor = textCursor();
    QString text = cursor.block().text();

    int firstNonSpace = 0;
    for (auto ch : text) {
        if (ch == ' ' || ch == '\t')
            ++firstNonSpace;
        else
            break;
    }

    int col = cursor.positionInBlock();
    auto mode = shift ? QTextCursor::KeepAnchor : QTextCursor::MoveAnchor;

    if (col == firstNonSpace || firstNonSpace >= text.length()) {
        cursor.movePosition(QTextCursor::StartOfBlock, mode);
    } else {
        cursor.movePosition(QTextCursor::StartOfBlock, mode);
        cursor.movePosition(QTextCursor::Right, mode, firstNonSpace);
    }
    setTextCursor(cursor);
}

void CustomTextEdit::keyPressEvent(QKeyEvent *event) {
    // Ctrl+D: duplicate line
    if (event->key() == Qt::Key_D && (event->modifiers() & Qt::ControlModifier)) {
        duplicateLine();
        return;
    }

    // Alt+Up: move line up
    if (event->key() == Qt::Key_Up && (event->modifiers() & Qt::AltModifier)) {
        moveLineUp();
        return;
    }

    // Alt+Down: move line down
    if (event->key() == Qt::Key_Down && (event->modifiers() & Qt::AltModifier)) {
        moveLineDown();
        return;
    }

    // Ctrl+/: toggle comment
    if (event->key() == Qt::Key_Slash && (event->modifiers() & Qt::ControlModifier)) {
        toggleComment();
        return;
    }

    // Home: smart home (first non-whitespace or column 0)
    if (event->key() == Qt::Key_Home && !(event->modifiers() & Qt::ControlModifier)) {
        smartHome(event->modifiers() & Qt::ShiftModifier);
        return;
    }

    // Tab: indent selection or insert spaces
    if (event->key() == Qt::Key_Tab) {
        if (hasMultiLineSelection()) {
            indentSelection();
        } else {
            QTextCursor cursor = textCursor();
            cursor.insertText("    ");
        }
        return;
    }

    // Shift+Tab: unindent
    if (event->key() == Qt::Key_Backtab) {
        if (hasMultiLineSelection()) {
            unindentSelection();
        } else {
            QTextCursor cursor = textCursor();
            cursor.movePosition(QTextCursor::StartOfBlock, QTextCursor::MoveAnchor);
            QString blockText = cursor.block().text();
            int spaces = 0;
            for (auto ch : blockText) {
                if (ch == ' ' && spaces < 4)
                    ++spaces;
                else
                    break;
            }
            if (spaces > 0) {
                cursor.movePosition(QTextCursor::Right, QTextCursor::KeepAnchor, spaces);
                cursor.removeSelectedText();
                setTextCursor(cursor);
            }
        }
        return;
    }

    // Enter: auto-indent
    if (event->key() == Qt::Key_Return || event->key() == Qt::Key_Enter) {
        autoIndentNewLine();
        return;
    }

    // Remove an empty auto-created pair with one Backspace press.
    if (event->key() == Qt::Key_Backspace && event->modifiers() == Qt::NoModifier) {
        QTextCursor cursor = textCursor();
        if (!cursor.hasSelection() && cursor.position() > 0) {
            const QChar previous = document()->characterAt(cursor.position() - 1);
            const QChar next = document()->characterAt(cursor.position());
            const bool isPair = (isOpenBracket(previous) && matchingBracket(previous) == next) ||
                                ((previous == '"' || previous == '\'') && previous == next);
            if (isPair) {
                cursor.beginEditBlock();
                cursor.deletePreviousChar();
                cursor.deleteChar();
                cursor.endEditBlock();
                return;
            }
        }
    }

    const auto insertPair = [this](QChar opening, QChar closing) {
        QTextCursor cursor = textCursor();
        if (cursor.hasSelection()) {
            const int selectionStart = cursor.selectionStart();
            QString selectedText = cursor.selectedText();
            selectedText.replace(QChar::ParagraphSeparator, '\n');
            cursor.insertText(QString(opening) + selectedText + QString(closing));
            cursor.setPosition(selectionStart + 1);
            cursor.setPosition(selectionStart + 1 + selectedText.size(),
                               QTextCursor::KeepAnchor);
        } else {
            cursor.insertText(QString(opening) + QString(closing));
            cursor.movePosition(QTextCursor::Left);
        }
        setTextCursor(cursor);
    };

    // Auto-close brackets
    if (event->text() == "{") {
        insertPair('{', '}');
        return;
    }
    if (event->text() == "(") {
        insertPair('(', ')');
        return;
    }
    if (event->text() == "[") {
        insertPair('[', ']');
        return;
    }

    // Skip-over closing brackets if the next char matches
    if (event->text() == "}" || event->text() == ")" || event->text() == "]") {
        QTextCursor cursor = textCursor();
        QChar nextChar = document()->characterAt(cursor.position());
        if (nextChar == event->text().at(0)) {
            cursor.movePosition(QTextCursor::Right);
            setTextCursor(cursor);
            return;
        }
    }

    // Auto-close/skip-over quotes
    if (event->text() == "\"" || event->text() == "'") {
        QTextCursor cursor = textCursor();
        QChar quote = event->text().at(0);
        if (cursor.hasSelection()) {
            insertPair(quote, quote);
            return;
        }
        QChar nextChar = document()->characterAt(cursor.position());
        if (nextChar == quote) {
            cursor.movePosition(QTextCursor::Right);
            setTextCursor(cursor);
            return;
        }
        cursor.insertText(QString(quote) + QString(quote));
        cursor.movePosition(QTextCursor::Left);
        setTextCursor(cursor);
        return;
    }

    QPlainTextEdit::keyPressEvent(event);
}

TextEditor::TextEditor(QWidget *parent)
    : QDialog(parent), m_modified(false), m_textEdit(nullptr), m_highlighter(nullptr),
      m_statusBar(nullptr), m_lineColLabel(nullptr), m_fontSize(24) {
    init();
}

void TextEditor::setText(const QString &text) {
    m_textEdit->setPlainText(text);
    m_textEdit->document()->setModified(false);
    m_modified = false;
    updateWindowTitle();
}

void TextEditor::setFileName(const QString &filen) {
    filename = filen;
    updateWindowTitle();
}

QString TextEditor::fileName() const {
    return filename;
}

void TextEditor::revealLocation(int lineNumber, int columnNumber, int matchLength) {
    const QTextBlock block = m_textEdit->document()->findBlockByLineNumber(
        qMax(0, lineNumber - 1));
    if (!block.isValid())
        return;

    const int lineLength = qMax(0, block.length() - 1);
    const int column = qBound(0, columnNumber, lineLength);
    const int selectionLength = qBound(0, matchLength, lineLength - column);
    QTextCursor cursor(block);
    cursor.setPosition(block.position() + column);
    if (selectionLength > 0) {
        cursor.setPosition(block.position() + column + selectionLength,
                           QTextCursor::KeepAnchor);
    }
    m_textEdit->setTextCursor(cursor);
    m_textEdit->centerCursor();
    m_textEdit->setFocus();
}

void TextEditor::updateWindowTitle() {
    QString title = "ACMX2";
    if (!filename.isEmpty()) {
        title += " - " + QFileInfo(filename).fileName();
    }
    if (m_modified) {
        title += " *";
    }
    setWindowTitle(title);
}

void TextEditor::init() {
    m_modified = false;
    QSettings editorSettings("LostSideDead");
    m_fontSize = qBound(8, editorSettings.value("editor/fontSize", 24).toInt(), 72);
    acmx2::applyCustomStyleIfEnabled(this);

    QVBoxLayout *layout = new QVBoxLayout(this);

    QMenuBar *menuBar = new QMenuBar(this);
#ifdef Q_OS_MACOS
    // Keep editor actions attached to each editor window, matching the main
    // interface, instead of moving them into macOS's global menu bar.
    menuBar->setNativeMenuBar(false);
#endif

    QMenu *fileMenu = menuBar->addMenu("&File");

    QAction *saveAction = fileMenu->addAction("&Save");
    saveAction->setShortcut(QKeySequence::Save);

    QAction *saveAsAction = fileMenu->addAction("Save &As...");
    saveAsAction->setShortcut(QKeySequence::SaveAs);

    fileMenu->addSeparator();

    QAction *closeAction = fileMenu->addAction("&Close");
    closeAction->setShortcut(QKeySequence::Close);

    QMenu *editMenu = menuBar->addMenu("&Edit");

    QAction *undoAction = editMenu->addAction("&Undo");
    undoAction->setShortcut(QKeySequence::Undo);

    QAction *redoAction = editMenu->addAction("&Redo");
    redoAction->setShortcut(QKeySequence::Redo);

    editMenu->addSeparator();

    QAction *cutAction = editMenu->addAction("Cu&t");
    cutAction->setShortcut(QKeySequence::Cut);

    QAction *copyAction = editMenu->addAction("&Copy");
    copyAction->setShortcut(QKeySequence::Copy);

    QAction *pasteAction = editMenu->addAction("&Paste");
    pasteAction->setShortcut(QKeySequence::Paste);

    editMenu->addSeparator();

    QAction *selectAllAction = editMenu->addAction("Select &All");
    selectAllAction->setShortcut(QKeySequence::SelectAll);

    editMenu->addSeparator();

    QAction *findAction = editMenu->addAction("&Find...");
    findAction->setShortcut(QKeySequence::Find);

    QAction *findNextAction = editMenu->addAction("Find &Next");
    findNextAction->setShortcut(QKeySequence::FindNext);

    QAction *findPrevAction = editMenu->addAction("Find Pre&vious");
    findPrevAction->setShortcut(QKeySequence::FindPrevious);

    QAction *replaceAction = editMenu->addAction("&Replace...");
    replaceAction->setShortcut(QKeySequence::Replace);

    editMenu->addSeparator();

    QAction *gotoLineAction = editMenu->addAction("&Go to Line...");
    gotoLineAction->setShortcut(QKeySequence(Qt::CTRL | Qt::Key_G));

    editMenu->addSeparator();

    QAction *duplicateAction = editMenu->addAction("&Duplicate Line");
    duplicateAction->setShortcut(QKeySequence(Qt::CTRL | Qt::Key_D));

    QAction *toggleCommentAction = editMenu->addAction("Toggle Co&mment");
    toggleCommentAction->setShortcut(QKeySequence(Qt::CTRL | Qt::Key_Slash));

    QAction *moveUpAction = editMenu->addAction("Move Line &Up");
    moveUpAction->setShortcut(QKeySequence(Qt::ALT | Qt::Key_Up));

    QAction *moveDownAction = editMenu->addAction("Move Line Dow&n");
    moveDownAction->setShortcut(QKeySequence(Qt::ALT | Qt::Key_Down));

    editMenu->addSeparator();

    QAction *shiftRightAction = editMenu->addAction("Shift &Right");
    shiftRightAction->setShortcut(QKeySequence(Qt::CTRL | Qt::Key_BracketRight));

    QAction *shiftLeftAction = editMenu->addAction("Shift &Left");
    shiftLeftAction->setShortcut(QKeySequence(Qt::CTRL | Qt::Key_BracketLeft));

    QMenu *viewMenu = menuBar->addMenu("&View");

    QAction *increaseFontAction = viewMenu->addAction("Increase Font Size");
    increaseFontAction->setShortcut(QKeySequence::ZoomIn);

    QAction *decreaseFontAction = viewMenu->addAction("Decrease Font Size");
    decreaseFontAction->setShortcut(QKeySequence::ZoomOut);

    QAction *resetFontAction = viewMenu->addAction("Reset Font Size");
    resetFontAction->setShortcut(QKeySequence(Qt::CTRL | Qt::Key_0));

    viewMenu->addSeparator();

    QAction *toggleWordWrapAction = viewMenu->addAction("Word Wrap");
    toggleWordWrapAction->setCheckable(true);
    toggleWordWrapAction->setChecked(
        editorSettings.value("editor/wordWrap", false).toBool());

    layout->setMenuBar(menuBar);

    m_textEdit = new CustomTextEdit(this);
    m_textEdit->setLineWrapMode(toggleWordWrapAction->isChecked()
                                    ? QPlainTextEdit::WidgetWidth
                                    : QPlainTextEdit::NoWrap);
    m_textEdit->setTabStopDistance(4 * m_textEdit->fontMetrics().horizontalAdvance(' '));
    updateFontSize();

    layout->addWidget(m_textEdit);

    m_statusBar = new QStatusBar(this);
    m_lineColLabel = new QLabel("Line: 1, Col: 1", this);
    m_statusBar->addPermanentWidget(m_lineColLabel);
    layout->addWidget(m_statusBar);

    m_highlighter = new GlslSyntaxHighlighter(m_textEdit->document());

    setLayout(layout);
    if (!restoreGeometry(editorSettings.value("editor/geometry").toByteArray()))
        setGeometry(300, 300, 1024, 768);

    saveAction->setEnabled(false);
    undoAction->setEnabled(false);
    redoAction->setEnabled(false);
    cutAction->setEnabled(false);
    copyAction->setEnabled(false);
    pasteAction->setEnabled(QApplication::clipboard()->mimeData()->hasText());

    connect(saveAction, &QAction::triggered, this, &TextEditor::saveContents);
    connect(saveAsAction, &QAction::triggered, this, &TextEditor::saveAs);
    connect(closeAction, &QAction::triggered, this, &TextEditor::close);

    connect(undoAction, &QAction::triggered, m_textEdit, &QPlainTextEdit::undo);
    connect(redoAction, &QAction::triggered, m_textEdit, &QPlainTextEdit::redo);
    connect(cutAction, &QAction::triggered, m_textEdit, &QPlainTextEdit::cut);
    connect(copyAction, &QAction::triggered, m_textEdit, &QPlainTextEdit::copy);
    connect(pasteAction, &QAction::triggered, m_textEdit, &QPlainTextEdit::paste);
    connect(selectAllAction, &QAction::triggered, m_textEdit, &QPlainTextEdit::selectAll);
    connect(duplicateAction, &QAction::triggered, m_textEdit, &CustomTextEdit::duplicateLine);
    connect(toggleCommentAction, &QAction::triggered, m_textEdit, &CustomTextEdit::toggleComment);
    connect(moveUpAction, &QAction::triggered, m_textEdit, &CustomTextEdit::moveLineUp);
    connect(moveDownAction, &QAction::triggered, m_textEdit, &CustomTextEdit::moveLineDown);

    connect(findAction, &QAction::triggered, this, &TextEditor::findText);
    connect(findNextAction, &QAction::triggered, this, &TextEditor::findNext);
    connect(findPrevAction, &QAction::triggered, this, &TextEditor::findPrevious);
    connect(replaceAction, &QAction::triggered, this, &TextEditor::replaceText);
    connect(gotoLineAction, &QAction::triggered, this, &TextEditor::gotoLine);

    connect(increaseFontAction, &QAction::triggered, this, &TextEditor::increaseFontSize);
    connect(decreaseFontAction, &QAction::triggered, this, &TextEditor::decreaseFontSize);
    connect(resetFontAction, &QAction::triggered, this, &TextEditor::resetFontSize);

    connect(toggleWordWrapAction, &QAction::triggered, this, [this](bool checked) {
        m_textEdit->setLineWrapMode(checked ? QPlainTextEdit::WidgetWidth : QPlainTextEdit::NoWrap);
        QSettings("LostSideDead").setValue("editor/wordWrap", checked);
    });

    connect(shiftRightAction, &QAction::triggered, this, [this]() {
        m_textEdit->indentSelection();
    });
    connect(shiftLeftAction, &QAction::triggered, this, [this]() {
        m_textEdit->unindentSelection();
    });

    connect(m_textEdit->document(), &QTextDocument::modificationChanged,
            this, [this, saveAction](bool modified) {
                m_modified = modified;
                saveAction->setEnabled(modified);
                updateWindowTitle();
            });

    connect(m_textEdit, &QPlainTextEdit::cursorPositionChanged, this, &TextEditor::updateCursorPosition);
    connect(m_textEdit, &QPlainTextEdit::copyAvailable, cutAction, &QAction::setEnabled);
    connect(m_textEdit, &QPlainTextEdit::copyAvailable, copyAction, &QAction::setEnabled);
    connect(m_textEdit, &QPlainTextEdit::undoAvailable, undoAction, &QAction::setEnabled);
    connect(m_textEdit, &QPlainTextEdit::redoAvailable, redoAction, &QAction::setEnabled);
    connect(m_textEdit->document(), &QTextDocument::blockCountChanged,
            this, [this](int) { updateCursorPosition(); });
    connect(QApplication::clipboard(), &QClipboard::dataChanged, this, [pasteAction]() {
        pasteAction->setEnabled(QApplication::clipboard()->mimeData()->hasText());
    });
    setAttribute(Qt::WA_DeleteOnClose);
}

void TextEditor::saveContents() {
    if (filename.isEmpty()) {
        saveAs();
        return;
    }

    writeFile(filename);
}

bool TextEditor::writeFile(const QString &filePath) {
    QSaveFile file(filePath);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        QMessageBox::warning(
            this, "Error",
            "Could not save file: " + filePath + "\n\n" + file.errorString());
        return false;
    }

    QTextStream out(&file);
    out << m_textEdit->toPlainText();
    out.flush();
    if (out.status() != QTextStream::Ok) {
        file.cancelWriting();
        QMessageBox::warning(this, "Error", "Could not write file: " + filePath);
        return false;
    }
    if (!file.commit()) {
        QMessageBox::warning(
            this, "Error",
            "Could not finish saving file: " + filePath + "\n\n" + file.errorString());
        return false;
    }

    filename = filePath;
    m_textEdit->document()->setModified(false);
    m_modified = false;
    updateWindowTitle();
    m_statusBar->showMessage("Saved " + QFileInfo(filename).fileName(), 2000);
    emit fileSaved(filename);
    return true;
}

void TextEditor::saveAs() {
    QSettings appSettings("LostSideDead");
    QString lastDir = appSettings.value("lastEditorSaveDir", QFileInfo(filename).absolutePath()).toString();
    QString newFileName = QFileDialog::getSaveFileName(
        this, "Save File As", lastDir + "/" + QFileInfo(filename).fileName(), "GLSL Files (*.glsl *.frag *.vert);;All Files (*)");

    if (!newFileName.isEmpty()) {
        if (writeFile(newFileName)) {
            appSettings.setValue("lastEditorSaveDir", QFileInfo(newFileName).absolutePath());
        }
    }
}

void TextEditor::findText() {
    bool ok;
    QString initialText = m_lastSearchText;
    const QString selectedText = m_textEdit->textCursor().selectedText();
    if (!selectedText.isEmpty() && !selectedText.contains(QChar::ParagraphSeparator))
        initialText = selectedText;
    QString searchText = QInputDialog::getText(this, "Find", "Enter text to find:",
                                               QLineEdit::Normal, initialText, &ok);
    if (ok && !searchText.isEmpty()) {
        m_lastSearchText = searchText;
        findNext();
    }
}

void TextEditor::findNext() {
    if (m_lastSearchText.isEmpty()) {
        findText();
        return;
    }

    QTextCursor cursor = m_textEdit->textCursor();
    QTextDocument::FindFlags flags;

    QTextCursor found = m_textEdit->document()->find(m_lastSearchText, cursor, flags);

    if (found.isNull()) {

        cursor.movePosition(QTextCursor::Start);
        found = m_textEdit->document()->find(m_lastSearchText, cursor, flags);

        if (found.isNull()) {
            m_statusBar->showMessage("Text not found: " + m_lastSearchText, 3000);
            return;
        }
    }

    m_textEdit->setTextCursor(found);
    m_statusBar->showMessage("Found: " + m_lastSearchText, 2000);
}

void TextEditor::findPrevious() {
    if (m_lastSearchText.isEmpty()) {
        findText();
        return;
    }

    QTextCursor cursor = m_textEdit->textCursor();
    QTextDocument::FindFlags flags = QTextDocument::FindBackward;

    QTextCursor found = m_textEdit->document()->find(m_lastSearchText, cursor, flags);

    if (found.isNull()) {
        cursor.movePosition(QTextCursor::End);
        found = m_textEdit->document()->find(m_lastSearchText, cursor, flags);

        if (found.isNull()) {
            m_statusBar->showMessage("Text not found: " + m_lastSearchText, 3000);
            return;
        }
    }

    m_textEdit->setTextCursor(found);
    m_statusBar->showMessage("Found: " + m_lastSearchText, 2000);
}

void TextEditor::replaceText() {
    bool ok;
    QString searchText = QInputDialog::getText(this, "Replace", "Find:",
                                               QLineEdit::Normal, m_lastSearchText, &ok);
    if (!ok || searchText.isEmpty())
        return;

    QString replaceWith = QInputDialog::getText(this, "Replace", "Replace with:",
                                                QLineEdit::Normal, "", &ok);
    if (!ok)
        return;

    QMessageBox::StandardButton reply = QMessageBox::question(
        this, "Replace All",
        "Replace all occurrences of '" + searchText + "' with '" + replaceWith + "'?",
        QMessageBox::Yes | QMessageBox::No);

    if (reply == QMessageBox::Yes) {
        QString text = m_textEdit->toPlainText();
        int count = text.count(searchText);
        if (count > 0) {
            text.replace(searchText, replaceWith);
            QTextCursor cursor(m_textEdit->document());
            cursor.beginEditBlock();
            cursor.select(QTextCursor::Document);
            cursor.insertText(text);
            cursor.endEditBlock();
        }
        m_lastSearchText = searchText;
        m_statusBar->showMessage("Replaced " + QString::number(count) + " occurrence(s)", 3000);
    }
}

void TextEditor::gotoLine() {
    bool ok;
    int lineNumber = QInputDialog::getInt(this, "Go to Line", "Line number:",
                                          1, 1, m_textEdit->document()->blockCount(), 1, &ok);
    if (ok) {
        QTextCursor cursor(m_textEdit->document()->findBlockByLineNumber(lineNumber - 1));
        m_textEdit->setTextCursor(cursor);
        m_textEdit->centerCursor();
    }
}

void TextEditor::increaseFontSize() {
    m_fontSize += 2;
    if (m_fontSize > 72)
        m_fontSize = 72;
    updateFontSize();
    QSettings("LostSideDead").setValue("editor/fontSize", m_fontSize);
}

void TextEditor::decreaseFontSize() {
    m_fontSize -= 2;
    if (m_fontSize < 8)
        m_fontSize = 8;
    updateFontSize();
    QSettings("LostSideDead").setValue("editor/fontSize", m_fontSize);
}

void TextEditor::resetFontSize() {
    m_fontSize = 24;
    updateFontSize();
    QSettings("LostSideDead").setValue("editor/fontSize", m_fontSize);
}

void TextEditor::updateFontSize() {
    QString styleSheet;
    if (acmx2::isCustomStyleEnabled()) {
        styleSheet = QString(
                         "QPlainTextEdit { "
                         "font-size: %1px; "
                         "font-family: 'Courier New', Courier, monospace; "
                         "}")
                         .arg(m_fontSize);
    } else {
        styleSheet = QString(
                         "QPlainTextEdit { "
                         "color: white; "
                         "font-size: %1px; "
                         "font-family: 'Courier New', Courier, monospace; "
                         "background-color: black; "
                         "}")
                         .arg(m_fontSize);
    }

    m_textEdit->setStyleSheet(styleSheet);
    m_textEdit->setTabStopDistance(4 * m_textEdit->fontMetrics().horizontalAdvance(' '));
    m_textEdit->updateLineNumberAreaWidth(0);
}

void TextEditor::updateCursorPosition() {
    QTextCursor cursor = m_textEdit->textCursor();
    int line = cursor.blockNumber() + 1;
    int col = cursor.columnNumber() + 1;
    QString status = QString("Line: %1, Col: %2 | Lines: %3")
                         .arg(line)
                         .arg(col)
                         .arg(m_textEdit->document()->blockCount());
    if (cursor.hasSelection())
        status += QString(" | Selected: %1").arg(cursor.selectionEnd() - cursor.selectionStart());
    m_lineColLabel->setText(status);
}

void TextEditor::closeEvent(QCloseEvent *event) {
    if (maybePromptSave()) {
        QSettings("LostSideDead").setValue("editor/geometry", saveGeometry());
        event->accept();
    } else {
        event->ignore();
    }
}

void TextEditor::keyPressEvent(QKeyEvent *event) {
    if (event->key() == Qt::Key_Escape && event->modifiers() == Qt::NoModifier) {
        if (maybePromptSave()) {
            QSettings("LostSideDead").setValue("editor/geometry", saveGeometry());
            accept();
        }
        event->accept();
        return;
    }
    QDialog::keyPressEvent(event);
}

bool TextEditor::maybePromptSave() {
    if (!m_modified)
        return true;
    QMessageBox::StandardButton reply = QMessageBox::question(
        this, "Unsaved Changes",
        "The document has been modified. Do you want to save your changes?",
        QMessageBox::Save | QMessageBox::Discard | QMessageBox::Cancel);

    if (reply == QMessageBox::Save) {
        saveContents();
        // saveContents clears m_modified on success; if the user cancelled
        // a Save As dialog the flag stays true, so honour that as Cancel.
        return !m_modified;
    }
    if (reply == QMessageBox::Discard)
        return true;
    return false;
}
