![image](/.Images/DS_Avatar_with_Logo.png =750x)

---

# How to use the R/Python script template


[[_TOC_]]

---

# Links

- [Script Template](https://dev.azure.com/DwrCymru/Data%20Team/_git/Data%20Science%20Projects%20Template?path=%2FCode%2FScript.txt)
- [Prject Folder Template](https://dev.azure.com/DwrCymru/Data%20Team/_git/Data%20Science%20Projects%20Template)


# Concept
The template script is specifically designed for the purpose of extracting three markdown documents from each code script in a given project.
The method of extraction is to use regular expressions to match certain patterns in the comments of the code.
These patterns are called 'triggers' and each document is comprised of the text following certain combinations of triggers.
This is so that no duplication of effort occurs when documenting vital project information. 

# Triggers
Triggers are used to assign meaning to certain lines of the code script, and they are used to specify which lines should be extracted to each output document.

1. `#____#` = Script Description
1. `#___#` = Script Header
1. `#__#`= Section Headers and Review Decision
1. `#_#` = Any other extractable comments
1. `#` = code comments that **will not** be extracted

# The three output documents
1. **readme**\
This is a High level overview of the script.
It is made up of the script header and script description.

1. **Decisions**\
This is a high level overview of the review decisions,
it is made up of the script header, section headers, and review decisions.

1. **Script Steps**\
This is the script without the code. It serves to show the steps taken in the script without the distraction of the code itself. If the code script is commented well, this document will be the starting point for the content added to the project's technical document.

# The template
Copy and paste the template below into the script you wish to work on. The header and descriptions need to appear once at the start of the script. The section title needs to appear at the begining of each code section. Remember to increase the numeral at the begining of the section title for each new section.
``` python
#___#---------------------------------------------------------------------------
#___#
#___#**Project:**         \
#___#**Script:**          \
#___#**Author:**          \
#___#**Date Created:**    \
#___#**Reviewer:**        \
#___#**Devops Feature:**  \
#___#**Devops Backlog:**  \
#___#**Devops Task:**     \
#___#**Devops Repo:**     \
#___#**MARS:**            
#___#
#___#
#____#Description
#____#
#____#
#___#
#___#---------------------------------------------------------------------------


#-------------------------------------------------------------------------------
#_#
#__#1.
#_#
#__#Review Decision:
#_#
#_#Author Notes\
#_#
#_#
#_#Reviewer Notes\
#_#
#_#
#_#Steps\

```

# Filling out the template
## The header

To fill out the template's header, simply type out each line's corresponding pieces of information at the end of each line but **before the backslash**.

``` python
#___#---------------------------------------------------------------------------
#___#
#___#**Project:**         Tutorial on how to fill in the template script\
#___#**Script:**          example.py\
#___#**Author:**          James Buntwal\
#___#**Date Created:**    September 2020\
#___#**Reviewer:**        TBC\
#___#**Devops Feature:**  #12345\
#___#**Devops Backlog:**  #12345\
#___#**Devops Task:**     #12345\
#___#**Devops Repo:**     Data-Science-How-Tos\
#___#**MARS:**            "S:\..."
#___#

```


## The description

To fill out the description, start writing on the line below the word description and after the description trigger. Try to keep the linewidth to roughly 80 characters, and remember to add a description trigger at the start of each line you want to include in the description.

``` python
#___#
#____#Description
#____#Begin writing your description here. Try to keep your linewidths to roughly
#____# 80 characters and add new description triggers for each new line. Also bear 
#____#in mind that when the description is extracted to markdown all of the lines 
#____#will collapse into a paragraph, so make sure to put a space at the start or 
#____#end of each line (not both).
#____#
#___#
#___#---------------------------------------------------------------------------

```


## The code sections
A core idea of this template is that the code we write should be split into small, consumable chunks. This is so that the readability of the code increases, making the review process easier and smoother.
When filling out the section header, put a descriptive title after the number and full stop, with a space between the full stop and title. For subsequent section headers you will need to increase the digit accordingly.
To add descriptive notes in the author notes section, write on the line after "Author Notes", and remember to add new triggers to each new line needed.
The code itself will go under the "Steps" line. Any code comments that begin with the `#_#` trigger will be collapsed into a paragraph when the steps section is extracted to a markdown document, so when coding, remember to include the trigger if you want the comment extracted, and use proper sentences and grammer.
Add a new section header for chunk of code in the script.


``` python
#-------------------------------------------------------------------------------
#_#
#__#1. Tutorial for filling out section header
#_#
#__#Review Decision:
#_#
#_#Author Notes\
#_#This is where the author can write descriptive notes about this particular
#_# code chunk. It is beneficial to the reviewer if this section gives plenty of
#_# detail about what is going on in the code chunk and why, what are the expected
#_# outcomes of this code chunk, and any other important details.
#_#Reviewer Notes\
#_#
#_#
#_#Steps\

#_#The first step is to assign the value 5 to the variable x.
x = 5

#_#The second step is to create a variable y which will be a list of integers from 0 to 10.
y = list(range(10))

#-------------------------------------------------------------------------------
#_#
#__#2. Second code chunk in the section header tutorial
#_#
#__#Review Decision:
#_#
#_#Author Notes\
#_#This is where the author can write descriptive notes about this particular
#_# code chunk. It is beneficial to the reviewer if this section gives plenty of
#_# detail about what is going on in the code chunk and why, what are the expected
#_# outcomes of this code chunk, and any other important details.
#_#Reviewer Notes\
#_#
#_#
#_#Steps\

#_#Print out the variables assigned above.
print(f"x = {x} and y = {y}")

#_#Put x and y into a dictionary called variable_dictionary.
variable_dictionary = { "x" : x,
                        "y" : y
                        }
```










