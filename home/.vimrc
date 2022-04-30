" __   _( ) __ ___ 
" \ \ / / | '_ ` _ \
"  \ V /| | | | | | |
"   \_/ |_|_| |_| |_|


"--- General
set ruler
set smartindent
set expandtab
set tabstop=4
set shiftwidth=4
set virtualedit=onemore
set autoread
set clipboard=unnamed,autoselect
set noswapfile
filetype plugin indent on
inoremap <silent> jj <ESC>
autocmd BufNewFile,BufRead *.launch setf xml

"--- Visual
set number
set display=lastline
syntax enable
colorscheme industry

"--- Movement
"set whichwrap=b,s,h,l,<,>,[,],~

"--- Search
set hlsearch
set wrapscan
set ignorecase
set smartcase
set incsearch
nnoremap <ESC><ESC> :nohl<CR>

"--- Add dein
"if &compatible
"  set nocompatible " Be iMproved
"endif
"
"set runtimepath+=~/.vim/dein/repos/github.com/Shougo/dein.vim
"
"if dein#load_state('~/.vim/dein')
"  call dein#begin('~/.vim/dein')
"
"  let s:toml = '~/.vim/dein/dein.toml'
"  call dein#load_toml(s:toml, {'lazy':0})
"
"  " Let dein manage dein
"  "call dein#add('~/.vim/dein/repos/github.com/Shougo/dein.vim')
"
"  call dein#end()
"  call dein#save_state()
"endif


" If you want to install not installed plugins on startup.
"if dein#check_install()
"  call dein#install()
"endif
