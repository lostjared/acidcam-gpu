#include "editor.hpp"
#include "custom_style.hpp"
#include <QAbstractItemView>
#include <QAction>
#include <QApplication>
#include <QCheckBox>
#include <QClipboard>
#include <QColor>
#include <QCompleter>
#include <QDoubleSpinBox>
#include <QFile>
#include <QFileDialog>
#include <QFont>
#include <QHBoxLayout>
#include <QInputDialog>
#include <QKeyEvent>
#include <QLabel>
#include <QMenuBar>
#include <QMessageBox>
#include <QMimeData>
#include <QMouseEvent>
#include <QPainter>
#include <QPalette>
#include <QPlainTextEdit>
#include <QPushButton>
#include <QRegularExpression>
#include <QSaveFile>
#include <QScrollArea>
#include <QScrollBar>
#include <QSettings>
#include <QSignalBlocker>
#include <QSizePolicy>
#include <QSlider>
#include <QSplitter>
#include <QStatusBar>
#include <QStringListModel>
#include <QSyntaxHighlighter>
#include <QTextBlock>
#include <QTextCharFormat>
#include <QTextCursor>
#include <QTextStream>
#include <QTimer>
#include <QVBoxLayout>

// --- CustomTextEdit ---

static QColor currentLineBackground(const QPalette &palette) {
    const QColor base = palette.color(QPalette::Base);
    const QColor text = palette.color(QPalette::Text);
    const qreal textMix = base.lightnessF() > 0.5 ? 0.08 : 0.15;

    auto blendChannel = [textMix](int baseChannel, int textChannel) {
        return qRound(baseChannel + (textChannel - baseChannel) * textMix);
    };

    return QColor(blendChannel(base.red(), text.red()),
                  blendChannel(base.green(), text.green()),
                  blendChannel(base.blue(), text.blue()));
}

static QColor diagnosticColor(ShaderDiagnosticSeverity severity) {
    switch (severity) {
    case ShaderDiagnosticSeverity::Warning:
        return QColor(240, 178, 50);
    case ShaderDiagnosticSeverity::Note:
        return QColor(80, 165, 230);
    case ShaderDiagnosticSeverity::Error:
        return QColor(225, 65, 65);
    }
    return QColor(225, 65, 65);
}

CustomTextEdit::CustomTextEdit(QWidget *parent) : QPlainTextEdit(parent) {
    m_lineNumberArea = new LineNumberArea(this);
    m_completionModel = new QStringListModel(this);
    m_completer = new QCompleter(m_completionModel, this);
    m_completer->setWidget(this);
    m_completer->setCompletionMode(QCompleter::PopupCompletion);
    m_completer->setCaseSensitivity(Qt::CaseInsensitive);
    m_completer->setFilterMode(Qt::MatchStartsWith);
    connect(m_completer,
            static_cast<void (QCompleter::*)(const QString &)>(
                &QCompleter::activated),
            this, &CustomTextEdit::insertCompletion);

    connect(this, &QPlainTextEdit::blockCountChanged, this, &CustomTextEdit::updateLineNumberAreaWidth);
    connect(this, &QPlainTextEdit::updateRequest, this, &CustomTextEdit::updateLineNumberArea);
    connect(this, &QPlainTextEdit::cursorPositionChanged, this, &CustomTextEdit::highlightCurrentLine);
    connect(this, &QPlainTextEdit::cursorPositionChanged, this, &CustomTextEdit::matchBrackets);

    updateLineNumberAreaWidth(0);
    highlightCurrentLine();
}

void CustomTextEdit::changeEvent(QEvent *event) {
    QPlainTextEdit::changeEvent(event);

    if (event->type() == QEvent::PaletteChange || event->type() == QEvent::StyleChange ||
        event->type() == QEvent::ApplicationPaletteChange) {
        matchBrackets();
        m_lineNumberArea->update();
        emit themeChanged();
    }
}

int CustomTextEdit::lineNumberAreaWidth() {
    int digits = 1;
    int max = qMax(1, blockCount());
    while (max >= 10) {
        max /= 10;
        ++digits;
    }
    digits = qMax(digits, 3);
    return 24 + fontMetrics().horizontalAdvance(QLatin1Char('9')) * digits;
}

void CustomTextEdit::setDiagnostics(
    const QVector<ShaderDiagnostic> &diagnostics) {
    m_diagnostics = diagnostics;
    m_lineNumberArea->update();
    matchBrackets();
}

void CustomTextEdit::setCompletionWords(const QStringList &words) {
    QStringList uniqueWords = words;
    uniqueWords.removeDuplicates();
    uniqueWords.sort(Qt::CaseInsensitive);
    m_completionModel->setStringList(uniqueWords);
}

QString CustomTextEdit::completionPrefix() const {
    QTextCursor cursor = textCursor();
    const int end = cursor.position();
    int start = end;
    while (start > 0) {
        const QChar character = document()->characterAt(start - 1);
        if (!character.isLetterOrNumber() && character != QLatin1Char('_') &&
            character != QLatin1Char('.')) {
            break;
        }
        --start;
    }
    cursor.setPosition(start);
    cursor.setPosition(end, QTextCursor::KeepAnchor);
    return cursor.selectedText();
}

void CustomTextEdit::insertCompletion(const QString &completion) {
    QTextCursor cursor = textCursor();
    const QString prefix = completionPrefix();
    if (!prefix.isEmpty()) {
        cursor.setPosition(cursor.position() - prefix.size());
        cursor.setPosition(cursor.position() + prefix.size(),
                           QTextCursor::KeepAnchor);
    }
    cursor.insertText(completion);
    setTextCursor(cursor);
}

void CustomTextEdit::showCompletionPopup(bool forced) {
    const QString prefix = completionPrefix();
    if (!forced && prefix.size() < 3) {
        m_completer->popup()->hide();
        return;
    }
    m_completer->setCompletionPrefix(prefix);
    if (m_completer->completionCount() == 0) {
        m_completer->popup()->hide();
        return;
    }
    m_completer->popup()->setCurrentIndex(
        m_completer->completionModel()->index(0, 0));
    QRect popupRect = cursorRect();
    popupRect.setWidth(qMax(280, m_completer->popup()->sizeHintForColumn(0) +
                                     m_completer->popup()
                                         ->verticalScrollBar()
                                         ->sizeHint()
                                         .width()));
    m_completer->complete(popupRect);
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
            painter.drawText(14, top, m_lineNumberArea->width() - 19,
                             fontMetrics().height(),
                             Qt::AlignRight, number);
            for (const ShaderDiagnostic &diagnostic : m_diagnostics) {
                if (diagnostic.line != blockNumber + 1)
                    continue;
                painter.setBrush(diagnosticColor(diagnostic.severity));
                painter.setPen(Qt::NoPen);
                const int diameter = qMin(9, fontMetrics().height() - 2);
                painter.drawEllipse(2, top + (fontMetrics().height() - diameter) / 2,
                                    diameter, diameter);
                break;
            }
        }
        block = block.next();
        top = bottom;
        bottom = top + qRound(blockBoundingRect(block).height());
        ++blockNumber;
    }
}

void CustomTextEdit::highlightCurrentLine() {
    matchBrackets();
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
    lineSelection.format.setBackground(currentLineBackground(palette()));
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

    for (const ShaderDiagnostic &diagnostic : m_diagnostics) {
        const QTextBlock block =
            doc->findBlockByLineNumber(qMax(0, diagnostic.line - 1));
        if (!block.isValid())
            continue;

        const int lineLength = qMax(0, block.length() - 1);
        const int column = qBound(0, diagnostic.column, lineLength);
        QTextEdit::ExtraSelection diagnosticSelection;
        diagnosticSelection.cursor = QTextCursor(block);
        diagnosticSelection.cursor.setPosition(block.position() + column);
        diagnosticSelection.cursor.setPosition(
            block.position() + qMin(lineLength, column + qMax(1, lineLength - column)),
            QTextCursor::KeepAnchor);
        diagnosticSelection.format.setUnderlineStyle(
            QTextCharFormat::WaveUnderline);
        diagnosticSelection.format.setUnderlineColor(
            diagnosticColor(diagnostic.severity));
        diagnosticSelection.format.setToolTip(diagnostic.message);
        selections.append(diagnosticSelection);
    }

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
    if (m_completer->popup()->isVisible()) {
        switch (event->key()) {
        case Qt::Key_Enter:
        case Qt::Key_Return:
        case Qt::Key_Escape:
        case Qt::Key_Tab:
        case Qt::Key_Backtab:
            event->ignore();
            return;
        default:
            break;
        }
    }

    if (event->key() == Qt::Key_Space &&
        event->modifiers() == Qt::ControlModifier) {
        showCompletionPopup(true);
        return;
    }

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
    const bool canComplete =
        event->modifiers() == Qt::NoModifier ||
        event->modifiers() == Qt::ShiftModifier;
    if (canComplete &&
        (!event->text().isEmpty() || event->key() == Qt::Key_Backspace)) {
        showCompletionPopup(false);
    } else {
        m_completer->popup()->hide();
    }
}

void CustomTextEdit::mousePressEvent(QMouseEvent *event) {
    if (event->button() == Qt::LeftButton &&
        (event->modifiers() & Qt::ControlModifier)) {
        const QTextCursor cursor = cursorForPosition(event->pos());
        const QString line = cursor.block().text();
        const QRegularExpression includeExpression(
            QStringLiteral("^\\s*#\\s*include\\s*[<\"]([^>\"]+)[>\"]"));
        const QRegularExpressionMatch match = includeExpression.match(line);
        if (match.hasMatch()) {
            const int column = cursor.positionInBlock();
            const int start = match.capturedStart(1);
            const int end = match.capturedEnd(1);
            if (column >= start && column <= end) {
                emit includeRequested(match.captured(1));
                event->accept();
                return;
            }
        }
    }
    QPlainTextEdit::mousePressEvent(event);
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
    updateCompletionWords();
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

void TextEditor::setCompilePending() {
    m_diagnostics.clear();
    m_currentDiagnostic = -1;
    m_textEdit->setDiagnostics(m_diagnostics);
    updateDiagnosticActions();
    m_compileStatusLabel->setText("Compiling...");
    m_compileStatusLabel->setStyleSheet("color: #e5b84b;");
    m_compileStatusLabel->setToolTip(QString());
}

void TextEditor::setCompileResult(bool success,
                                  const QString &compilerOutput) {
    m_diagnostics = parseDiagnostics(compilerOutput);
    m_currentDiagnostic = -1;
    m_textEdit->setDiagnostics(m_diagnostics);
    updateDiagnosticActions();

    int errors = 0;
    int warnings = 0;
    for (const ShaderDiagnostic &diagnostic : m_diagnostics) {
        errors += diagnostic.severity == ShaderDiagnosticSeverity::Error;
        warnings += diagnostic.severity == ShaderDiagnosticSeverity::Warning;
    }
    if (success) {
        m_compileStatusLabel->setText(
            warnings > 0
                ? QString("Compiled successfully · %1 warning(s)").arg(warnings)
                : QStringLiteral("Compiled successfully"));
        m_compileStatusLabel->setStyleSheet(
            warnings > 0 ? "color: #e5b84b;" : "color: #55c878;");
        m_compileStatusLabel->setToolTip(compilerOutput.left(8192));
        m_statusBar->showMessage(
            warnings > 0 ? "Shader applied; press F8 to review warnings"
                         : "Shader compiled and applied",
            3000);
        return;
    }
    QString status = "Compile failed";
    if (!m_diagnostics.isEmpty()) {
        status += QString(" · %1 error(s), %2 warning(s)")
                      .arg(errors)
                      .arg(warnings);
    }
    m_compileStatusLabel->setText(status);
    m_compileStatusLabel->setStyleSheet("color: #e14b4b;");
    m_compileStatusLabel->setToolTip(compilerOutput.left(8192));
    m_statusBar->showMessage(
        m_diagnostics.isEmpty()
            ? "Compilation failed; see the ACMX log for compiler output"
            : "Press F8 to visit the first compiler diagnostic",
        5000);
}

void TextEditor::setShaderContext(
    bool acmxvk, const QVector<ShaderEditorUniform> &uniforms,
    const QString &libraryDirectory) {
    m_acmxvkContext = acmxvk;
    m_uniforms = uniforms;
    m_libraryDirectory = libraryDirectory;
    m_snippetsMenu->setEnabled(acmxvk);
    m_livePreviewCheck->setEnabled(!libraryDirectory.isEmpty());
    updateCompletionWords();
    rebuildUniformControls();
}

void TextEditor::setUniformValue(const QString &name, double value) {
    for (ShaderEditorUniform &uniform : m_uniforms) {
        if (uniform.name != name)
            continue;
        uniform.value = qBound(uniform.minimum, value, uniform.maximum);
        if (QDoubleSpinBox *spin = m_uniformSpins.value(name)) {
            const QSignalBlocker blocker(spin);
            spin->setValue(uniform.value);
        }
        if (QSlider *slider = m_uniformSliders.value(name)) {
            const QSignalBlocker blocker(slider);
            const double range = uniform.maximum - uniform.minimum;
            const int position = range > 0.0
                                     ? qRound((uniform.value - uniform.minimum) /
                                              range * slider->maximum())
                                     : 0;
            slider->setValue(position);
        }
        break;
    }
}

void TextEditor::openInclude(const QString &includeName) {
    const QFileInfo sourceInfo(filename);
    const QStringList candidates{
        sourceInfo.dir().filePath(includeName),
        QDir(m_libraryDirectory).filePath(includeName)};
    for (const QString &candidate : candidates) {
        const QFileInfo includeInfo(candidate);
        if (includeInfo.exists() && includeInfo.isFile()) {
            emit openFileRequested(includeInfo.canonicalFilePath(), 1);
            return;
        }
    }
    m_statusBar->showMessage(
        QStringLiteral("Include file not found: %1").arg(includeName), 4000);
}

void TextEditor::requestPreview() {
    if (filename.isEmpty() || m_libraryDirectory.isEmpty())
        return;
    emit previewRequested(filename, m_textEdit->toPlainText());
}

void TextEditor::revertContents() {
    if (filename.isEmpty())
        return;
    QFile file(filename);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        QMessageBox::warning(this, tr("Revert Shader"), file.errorString());
        return;
    }
    setText(QString::fromUtf8(file.readAll()));
    m_statusBar->showMessage(tr("Restored the saved shader"), 2500);
}

void TextEditor::rebuildUniformControls() {
    m_uniformSpins.clear();
    m_uniformSliders.clear();
    while (QLayoutItem *item = m_uniformLayout->takeAt(0)) {
        if (QWidget *widget = item->widget())
            widget->deleteLater();
        delete item;
    }

    const bool visible = m_acmxvkContext && !m_uniforms.isEmpty();
    m_uniformPanel->setVisible(visible);
    m_uniformScroll->setVisible(visible);
    if (!visible)
        return;

    auto *heading = new QLabel(tr("Custom Uniforms"), m_uniformPanel);
    QFont headingFont = heading->font();
    headingFont.setBold(true);
    heading->setFont(headingFont);
    m_uniformLayout->addWidget(heading);

    for (const ShaderEditorUniform &uniform : m_uniforms) {
        auto *nameLabel = new QLabel(uniform.name, m_uniformPanel);
        if (uniform.slot >= 0) {
            nameLabel->setToolTip(
                QStringLiteral("ext.custom_uniforms[%1].%2")
                    .arg(uniform.slot / 4)
                    .arg(QStringLiteral("xyzw").at(uniform.slot % 4)));
        }
        auto *slider = new QSlider(Qt::Horizontal, m_uniformPanel);
        slider->setRange(0, 1000);
        auto *spin = new QDoubleSpinBox(m_uniformPanel);
        spin->setDecimals(8);
        spin->setRange(uniform.minimum, uniform.maximum);
        spin->setSingleStep(uniform.step);
        spin->setValue(uniform.value);
        const double range = uniform.maximum - uniform.minimum;
        slider->setValue(range > 0.0
                             ? qRound((uniform.value - uniform.minimum) /
                                      range * slider->maximum())
                             : 0);
        m_uniformLayout->addWidget(nameLabel);
        m_uniformLayout->addWidget(slider);
        m_uniformLayout->addWidget(spin);
        m_uniformSpins.insert(uniform.name, spin);
        m_uniformSliders.insert(uniform.name, slider);

        connect(slider, &QSlider::valueChanged, this,
                [this, uniform, spin, slider](int position) {
                    const double ratio = slider->maximum() > 0
                                             ? static_cast<double>(position) /
                                                   slider->maximum()
                                             : 0.0;
                    const double raw = uniform.minimum +
                                       (uniform.maximum - uniform.minimum) * ratio;
                    const double steps = uniform.step > 0.0
                                             ? qRound((raw - uniform.minimum) /
                                                      uniform.step)
                                             : 0.0;
                    const double value = qBound(
                        uniform.minimum,
                        uniform.minimum + steps * uniform.step,
                        uniform.maximum);
                    const QSignalBlocker blocker(spin);
                    spin->setValue(value);
                    emit uniformValueChanged(uniform.name, value);
                });
        connect(spin,
                static_cast<void (QDoubleSpinBox::*)(double)>(
                    &QDoubleSpinBox::valueChanged),
                this, [this, uniform, slider](double value) {
                    const double range = uniform.maximum - uniform.minimum;
                    const int position = range > 0.0
                                             ? qRound((value - uniform.minimum) /
                                                      range * slider->maximum())
                                             : 0;
                    const QSignalBlocker blocker(slider);
                    slider->setValue(position);
                    emit uniformValueChanged(uniform.name, value);
                });
    }
    m_uniformLayout->addStretch();
}

void TextEditor::updateCompletionWords() {
    QStringList words{
        "break", "case", "const", "continue",
        "default", "discard", "do", "else",
        "false", "for", "if", "in",
        "inout", "layout", "out", "precision",
        "return", "struct", "switch", "true",
        "uniform", "while", "bool", "int",
        "uint", "float", "double", "vec2",
        "vec3", "vec4", "ivec2", "ivec3",
        "ivec4", "uvec2", "uvec3", "uvec4",
        "bvec2", "bvec3", "bvec4", "mat2",
        "mat3", "mat4", "sampler1D", "sampler2D",
        "sampler2DArray", "samplerCube", "image2D", "abs",
        "acos", "all", "any", "asin",
        "atan", "ceil", "clamp", "cos",
        "cross", "degrees", "distance", "dot",
        "exp", "exp2", "floor", "fract",
        "imageLoad", "imageSize", "imageStore", "length",
        "log", "log2", "max", "min",
        "mix", "mod", "normalize", "pow",
        "radians", "reflect", "refract", "round",
        "sign", "sin", "smoothstep", "sqrt",
        "step", "tan", "texelFetch", "texture",
        "textureSize", "transpose"};

    if (m_acmxvkContext) {
        words << "input_image"
              << "output_image"
              << "history"
              << "spectrum"
              << "spectrum_history"
              << "ext.mouse"
              << "ext.u0"
              << "ext.u1"
              << "ext.u2"
              << "ext.u3"
              << "ext.custom_uniforms"
              << "ext.audio_bands"
              << "ext.audio_history"
              << "pc.screen_width"
              << "pc.screen_height"
              << "pc.effects_on"
              << "pc.rotation_degrees"
              << "pc.params"
              << "iResolution"
              << "iTime"
              << "iTimeDelta"
              << "iFrame"
              << "amp_low"
              << "amp_mid"
              << "amp_high"
              << "gl_GlobalInvocationID"
              << "gl_LocalInvocationID"
              << "gl_LocalInvocationIndex"
              << "gl_NumWorkGroups"
              << "gl_WorkGroupID"
              << "gl_WorkGroupSize"
              << "gl_FragCoord";
        for (const ShaderEditorUniform &uniform : m_uniforms)
            words.append(uniform.name);
    }

    const QString source = m_textEdit->toPlainText();
    const QRegularExpression declarations(
        QStringLiteral("(?:#\\s*define|\\b(?:void|bool|int|uint|float|double|"
                       "[biu]?vec[234]|mat[234])\\s+)\\s*"
                       "([A-Za-z_][A-Za-z0-9_]*)"));
    QRegularExpressionMatchIterator matches = declarations.globalMatch(source);
    while (matches.hasNext())
        words.append(matches.next().captured(1));
    m_textEdit->setCompletionWords(words);
}

QString TextEditor::customUniformDefines() const {
    QString defines;
    static const QString components = QStringLiteral("xyzw");
    for (int index = 0; index < m_uniforms.size(); ++index) {
        const ShaderEditorUniform &uniform = m_uniforms[index];
        const int slot = uniform.slot >= 0 ? uniform.slot : index;
        defines += QString("#define %1 ext.custom_uniforms[%2].%3\n")
                       .arg(uniform.name)
                       .arg(slot / 4)
                       .arg(components.at(slot % 4));
    }
    return defines;
}

void TextEditor::insertSnippet(const QString &snippet) {
    if (snippet.isEmpty()) {
        m_statusBar->showMessage("The active library has no custom uniforms",
                                 3000);
        return;
    }
    QTextCursor cursor = m_textEdit->textCursor();
    QString insertion = snippet;
    if (cursor.positionInBlock() != 0)
        insertion.prepend(QLatin1Char('\n'));
    if (!insertion.endsWith(QLatin1Char('\n')))
        insertion.append(QLatin1Char('\n'));
    cursor.insertText(insertion);
    m_textEdit->setTextCursor(cursor);
    m_textEdit->setFocus();
    updateCompletionWords();
}

QVector<ShaderDiagnostic>
TextEditor::parseDiagnostics(const QString &compilerOutput) const {
    QVector<ShaderDiagnostic> diagnostics;
    const QFileInfo editedFile(filename);
    const QString editedCanonical = editedFile.canonicalFilePath();
    const QRegularExpression withColumn(
        QStringLiteral("^(.+):(\\d+):(\\d+):\\s*"
                       "(?:(error|warning|note)\\s*:\\s*)?(.*)$"),
        QRegularExpression::CaseInsensitiveOption);
    const QRegularExpression withoutColumn(
        QStringLiteral("^(.+):(\\d+):\\s*"
                       "(?:(error|warning|note)\\s*:\\s*)?(.*)$"),
        QRegularExpression::CaseInsensitiveOption);

    const QStringList lines = compilerOutput.split(QLatin1Char('\n'));
    for (const QString &line : lines) {
        QRegularExpressionMatch match = withColumn.match(line.trimmed());
        const bool hasColumn = match.hasMatch();
        if (!hasColumn)
            match = withoutColumn.match(line.trimmed());
        if (!match.hasMatch())
            continue;

        QString reportedPath = match.captured(1).trimmed();
        if (reportedPath.startsWith(QLatin1Char('"')) &&
            reportedPath.endsWith(QLatin1Char('"'))) {
            reportedPath = reportedPath.mid(1, reportedPath.size() - 2);
        }
        QFileInfo reportedFile(reportedPath);
        if (reportedFile.isRelative()) {
            reportedFile = QFileInfo(editedFile.absoluteDir(), reportedPath);
        }
        const QString reportedCanonical = reportedFile.canonicalFilePath();
        const bool sameFile =
            (!editedCanonical.isEmpty() && !reportedCanonical.isEmpty() &&
             editedCanonical == reportedCanonical) ||
            QFileInfo(reportedPath).fileName() == editedFile.fileName();
        if (!sameFile)
            continue;

        ShaderDiagnostic diagnostic;
        diagnostic.line = qMax(1, match.captured(2).toInt());
        diagnostic.column = hasColumn
                                ? qMax(0, match.captured(3).toInt() - 1)
                                : 0;
        const QString severity =
            match.captured(hasColumn ? 4 : 3).toLower();
        if (severity == QStringLiteral("warning"))
            diagnostic.severity = ShaderDiagnosticSeverity::Warning;
        else if (severity == QStringLiteral("note"))
            diagnostic.severity = ShaderDiagnosticSeverity::Note;
        const QString detail = match.captured(hasColumn ? 5 : 4).trimmed();
        diagnostic.message =
            detail.isEmpty() ? line.trimmed() : detail;
        diagnostics.append(diagnostic);
    }
    return diagnostics;
}

void TextEditor::navigateDiagnostic(int offset) {
    if (m_diagnostics.isEmpty())
        return;
    if (m_currentDiagnostic < 0)
        m_currentDiagnostic = offset < 0 ? m_diagnostics.size() - 1 : 0;
    else
        m_currentDiagnostic =
            (m_currentDiagnostic + offset + m_diagnostics.size()) %
            m_diagnostics.size();
    const ShaderDiagnostic &diagnostic = m_diagnostics[m_currentDiagnostic];
    revealLocation(diagnostic.line, diagnostic.column, 1);
    m_statusBar->showMessage(
        QString("Diagnostic %1 of %2: %3")
            .arg(m_currentDiagnostic + 1)
            .arg(m_diagnostics.size())
            .arg(diagnostic.message));
}

void TextEditor::updateDiagnosticActions() {
    const bool available = !m_diagnostics.isEmpty();
    m_nextDiagnosticAction->setEnabled(available);
    m_previousDiagnosticAction->setEnabled(available);
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
    setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
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

    QMenu *diagnosticsMenu = menuBar->addMenu("&Diagnostics");
    m_nextDiagnosticAction = diagnosticsMenu->addAction("&Next Diagnostic");
    m_nextDiagnosticAction->setShortcut(QKeySequence(Qt::Key_F8));
    m_previousDiagnosticAction =
        diagnosticsMenu->addAction("&Previous Diagnostic");
    m_previousDiagnosticAction->setShortcut(
        QKeySequence(Qt::SHIFT | Qt::Key_F8));

    m_snippetsMenu = menuBar->addMenu("&Snippets");
    QAction *engineStateSnippet =
        m_snippetsMenu->addAction("Input and Engine State Bindings");
    QAction *pushConstantsSnippet =
        m_snippetsMenu->addAction("Fragment Push Constants");
    QAction *historySnippet =
        m_snippetsMenu->addAction("Frame History Binding");
    QAction *audioSnippet =
        m_snippetsMenu->addAction("FFT and FFT History Bindings");
    QAction *uniformSnippet =
        m_snippetsMenu->addAction("Custom Uniform Defines");
    m_snippetsMenu->addSeparator();
    QAction *computeMainSnippet =
        m_snippetsMenu->addAction("Compute Output and Main Function");
    m_snippetsMenu->setEnabled(false);

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

    auto *previewBar = new QHBoxLayout();
    auto *previewButton = new QPushButton(tr("Compile Preview"), this);
    auto *saveApplyButton = new QPushButton(tr("Save && Apply"), this);
    auto *revertButton = new QPushButton(tr("Revert"), this);
    for (QPushButton *button : {previewButton, saveApplyButton, revertButton}) {
        button->setAutoDefault(false);
        button->setDefault(false);
    }
    m_livePreviewCheck = new QCheckBox(tr("Live Preview"), this);
    m_livePreviewCheck->setChecked(
        editorSettings.value("editor/livePreview", false).toBool());
    m_livePreviewCheck->setEnabled(false);
    previewBar->addWidget(previewButton);
    previewBar->addWidget(saveApplyButton);
    previewBar->addWidget(revertButton);
    previewBar->addStretch();
    previewBar->addWidget(m_livePreviewCheck);
    layout->addLayout(previewBar);

    auto *splitter = new QSplitter(Qt::Horizontal, this);
    splitter->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    m_textEdit = new CustomTextEdit(splitter);
    m_textEdit->setLineWrapMode(toggleWordWrapAction->isChecked()
                                    ? QPlainTextEdit::WidgetWidth
                                    : QPlainTextEdit::NoWrap);
    m_textEdit->setTabStopDistance(4 * m_textEdit->fontMetrics().horizontalAdvance(' '));
    updateFontSize();

    splitter->addWidget(m_textEdit);

    m_uniformScroll = new QScrollArea(splitter);
    m_uniformScroll->setWidgetResizable(true);
    m_uniformScroll->setMinimumWidth(220);
    m_uniformScroll->setMaximumWidth(360);
    m_uniformPanel = new QWidget(m_uniformScroll);
    m_uniformLayout = new QVBoxLayout(m_uniformPanel);
    m_uniformLayout->setAlignment(Qt::AlignTop);
    m_uniformScroll->setWidget(m_uniformPanel);
    splitter->addWidget(m_uniformScroll);
    splitter->setStretchFactor(0, 1);
    splitter->setStretchFactor(1, 0);
    m_uniformPanel->hide();
    m_uniformScroll->hide();
    layout->addWidget(splitter, 1);

    m_statusBar = new QStatusBar(this);
    m_statusBar->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    m_lineColLabel = new QLabel("Line: 1, Col: 1", this);
    m_compileStatusLabel = new QLabel("Not compiled", this);
    m_statusBar->addWidget(m_compileStatusLabel);
    m_statusBar->addPermanentWidget(m_lineColLabel);
    layout->addWidget(m_statusBar, 0);

    m_highlighter = new GlslSyntaxHighlighter(m_textEdit->document());
    m_highlighter->setEditorPalette(m_textEdit->palette());
    connect(m_textEdit, &CustomTextEdit::themeChanged, this, [this]() {
        m_highlighter->setEditorPalette(m_textEdit->palette());
    });

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
    connect(previewButton, &QPushButton::clicked, this,
            &TextEditor::requestPreview);
    connect(saveApplyButton, &QPushButton::clicked, this,
            &TextEditor::saveContents);
    connect(revertButton, &QPushButton::clicked, this,
            &TextEditor::revertContents);
    connect(m_livePreviewCheck, &QCheckBox::toggled, this,
            [this](bool checked) {
                QSettings("LostSideDead").setValue("editor/livePreview", checked);
                if (checked)
                    m_previewTimer->start();
            });

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
    connect(m_nextDiagnosticAction, &QAction::triggered, this,
            [this]() { navigateDiagnostic(1); });
    connect(m_previousDiagnosticAction, &QAction::triggered, this,
            [this]() { navigateDiagnostic(-1); });
    updateDiagnosticActions();
    connect(engineStateSnippet, &QAction::triggered, this, [this]() {
        insertSnippet(QStringLiteral(R"glsl(layout(set = 0, binding = 0) uniform sampler2D input_image;

layout(set = 0, binding = 1, std140) uniform SpriteExtended {
    vec4 mouse;
    vec4 u0;
    vec4 u1;
    vec4 u2;
    vec4 u3;
    vec4 custom_uniforms[16];
    vec4 audio_bands;
    vec4 audio_history;
} ext;

#define iResolution max(ext.u0.zw, vec2(1.0))
#define iTime ext.u2.y
#define iTimeDelta ext.u1.x
#define iFrame ext.u2.x
#define amp_low ext.audio_bands.x
#define amp_mid ext.audio_bands.y
#define amp_high ext.audio_bands.z)glsl"));
    });
    connect(pushConstantsSnippet, &QAction::triggered, this, [this]() {
        insertSnippet(QStringLiteral(R"glsl(layout(push_constant) uniform SpritePushConstants {
    float screen_width;
    float screen_height;
    float sprite_pos_x;
    float sprite_pos_y;
    float sprite_size_w;
    float sprite_size_h;
    float effects_on;
    float rotation_degrees;
    vec4 params;
} pc;)glsl"));
    });
    connect(historySnippet, &QAction::triggered, this, [this]() {
        insertSnippet(QStringLiteral(
            "layout(set = 0, binding = 2) uniform sampler2DArray history;"));
    });
    connect(audioSnippet, &QAction::triggered, this, [this]() {
        insertSnippet(QStringLiteral(R"glsl(layout(set = 0, binding = 3) uniform sampler1D spectrum;
layout(set = 0, binding = 4) uniform sampler1DArray spectrum_history;)glsl"));
    });
    connect(uniformSnippet, &QAction::triggered, this,
            [this]() { insertSnippet(customUniformDefines()); });
    connect(computeMainSnippet, &QAction::triggered, this, [this]() {
        insertSnippet(QStringLiteral(R"glsl(layout(set = 0, binding = 5, rgba8) writeonly uniform image2D output_image;

void main() {
    ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = imageSize(output_image);
    if (any(greaterThanEqual(pixel, size))) {
        return;
    }

    vec2 uv = (vec2(pixel) + vec2(0.5)) / vec2(size);
    vec4 source_color = texture(input_image, uv);
    imageStore(output_image, pixel, source_color);
})glsl"));
    });

    connect(m_textEdit->document(), &QTextDocument::modificationChanged,
            this, [this, saveAction](bool modified) {
                m_modified = modified;
                saveAction->setEnabled(modified);
                updateWindowTitle();
            });
    m_previewTimer = new QTimer(this);
    m_previewTimer->setSingleShot(true);
    m_previewTimer->setInterval(650);
    connect(m_previewTimer, &QTimer::timeout, this,
            &TextEditor::requestPreview);
    connect(m_textEdit->document(), &QTextDocument::contentsChanged, this,
            [this]() {
                updateCompletionWords();
                if (m_livePreviewCheck->isChecked())
                    m_previewTimer->start();
            });
    connect(m_textEdit, &CustomTextEdit::includeRequested, this,
            &TextEditor::openInclude);

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
        this, "Save File As", lastDir + "/" + QFileInfo(filename).fileName(), "GLSL Files (*.glsl *.frag *.vert *.comp);;All Files (*)");

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
