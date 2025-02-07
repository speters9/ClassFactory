
preamble = r"""

%----------------------------------------------------------------------------------------
%	PACKAGES AND THEMES
%----------------------------------------------------------------------------------------

\documentclass{beamer}

\mode<presentation> {

% The Beamer class comes with a number of default slide themes
% which change the colors and layouts of slides. Below this is a list
% of all the themes, uncomment each in turn to see what they look like.

%\usetheme{default}
%\usetheme{AnnArbor}
%\usetheme{Antibes}
%\usetheme{Bergen}
%\usetheme{Berkeley}
%\usetheme{Berlin}
%\usetheme{Boadilla}
%\usetheme{CambridgeUS}
%\usetheme{Copenhagen}
%\usetheme{Darmstadt}
%\usetheme{Dresden}
%\usetheme{Frankfurt}
%\usetheme{Goettingen}
%\usetheme{Hannover}
%\usetheme{Ilmenau}
%\usetheme{JuanLesPins}
%\usetheme{Luebeck}
\usetheme{Madrid}
%\usetheme{Malmoe}
%\usetheme{Marburg}
%\usetheme{Montpellier}
%\usetheme{PaloAlto}
%\usetheme{Pittsburgh}
%\usetheme{Rochester}
%\usetheme{Singapore}
%\usetheme{Szeged}
%\usetheme{Warsaw}

% As well as themes, the Beamer class has a number of color themes
% for any slide theme. Uncomment each of these in turn to see how it
% changes the colors of your current slide theme.

%\usecolortheme{albatross}
%\usecolortheme{beaver}
%\usecolortheme{beetle}
%\usecolortheme{crane}
%\usecolortheme{dolphin}
%\usecolortheme{dove}
%\usecolortheme{fly}
%\usecolortheme{lily}
%\usecolortheme{orchid}
%\usecolortheme{rose}
%\usecolortheme{seagull}
%\usecolortheme{seahorse}
%\usecolortheme{whale}
%\usecolortheme{wolverine}

%\setbeamertemplate{footline} % To remove the footer line in all slides uncomment this line
%\setbeamertemplate{footline}[page number] % To replace the footer line in all slides with a simple slide count uncomment this line

\setbeamertemplate{navigation symbols}{} % To remove the navigation symbols from the bottom of all slides uncomment this line
\usepackage[backend=biber,style=apa,citestyle=authoryear, natbib=true,labeldate=year]{biblatex}
    \DeclareLanguageMapping{english}{english-apa}
    \DefineBibliographyStrings{polish}{andothers={i inni},and={i}}

}

\setbeamertemplate{page number in head/foot}{\insertframenumber / \inserttotalframenumber}
\newcommand{\graycite}[1]{{\scriptsize \color{gray} #1}}

\usepackage{lmodern}
\usepackage{lipsum}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{rotating}
\usepackage{caption}
\usepackage{subcaption}
\usepackage{comment}
\usepackage{tikz}
\usepackage{makecell}
\usepackage{array}
\usepackage{color,soul}
\usetikzlibrary{positioning}

\newcolumntype{P}[1]{>{\hspace{0pt}}p{#1}}

\usepackage{siunitx}
\newcolumntype{d}{S[
    input-open-uncertainty=,
    input-close-uncertainty=,
    parse-numbers = false,
    table-align-text-pre=false,
    table-align-text-post=false
]}
\newcommand\blfootnote[1]{%
  \begingroup
  \renewcommand\thefootnote{}\footnote{#1}%
  \addtocounter{footnote}{-1}%
  \endgroup
}

\usepackage[english]{babel}

\setbeamertemplate{subcaption}{%
    \insertcaption\par
}
\setbeamertemplate{caption label separator}{}

\usepackage{stackengine}

\bibliography{Zotero}
\newenvironment{wideitemize}{\itemize\addtolength{\itemsep}{6pt}}{\enditemize}
\newenvironment{tightitemize}{
  \begin{itemize}
    \setlength{\itemsep}{0pt}%
    \setlength{\parskip}{0pt}%
    \setlength{\topsep}{0pt}%
}{\end{itemize}}

\renewcommand*{\nameyeardelim}{\addspace}


\AtBeginSection[]
  {
     \begin{frame}<beamer>
     \frametitle{Agenda}
     \tableofcontents[currentsection]
     \end{frame}
  }

%----------------------------------------------------------------------------------------
%	TITLE PAGE
%----------------------------------------------------------------------------------------
"""
