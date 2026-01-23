Here's an overview of the report structure

main 
  |
  --> parts
        |
        --> # appendix: self explanatory
        --> # commands: a file for all glossaries and our own commands
        --> # prelude:  a file for everything related to the printing format
        --> # preamble: contains all packages and settings
        --> # tools:    contains copies of figures and table design
        --> ch_*:       files for each chapter of the report      

I was quite a fight to get the LaTeX files to properly compile without packages clashing. 

I had to tweak the settings in the JSON file (settings.json). I saved a copy of the settings that finally worked. Try to paste them at the end of the json-file BEFORE compiling the thing. 
With the current settings, the report compiles every time it's saved. So, cmd+S will save and compile the files. We might have to change this later, as there might be issues with syncing to git. 

To end, you might have to install the "LaTeX" extension. I'll also recommend "LaTeX language support" and "LaTeX Workshop". 