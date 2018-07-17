# Used to enable bash autocomplete on `insar` commands
# Usage: 
#	. .insar-complete.sh
#
# To automatically enable, add the above line to your ~/.bashrc

_insar_completion() {
    COMPREPLY=( $( env COMP_WORDS="${COMP_WORDS[*]}" \
                   COMP_CWORD=$COMP_CWORD \
                   _INSAR_COMPLETE=complete $1 ) )
    return 0
}

complete -F _insar_completion -o default insar;
