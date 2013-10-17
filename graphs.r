#!/usr/bin/Rscript

# R options
options("width"=220)

##################################
## Functions
##################################

# Functions to concatenate two strings
p <- function(a, b) { return(paste(a, b, sep="")) }
d <- function(a, b) { return(paste(a, b, sep="/")) }

##################################
## GPU parameters
##################################

# Set the default clocks (GTX470 rounded)
defaultcore = 1200
defaultmem = 900

# Set the roofline intersection point (in ops/byte)
roofline_point = 5

##################################
## Settings
##################################

# Exclude benchmarks from the results
bench_excludes = c(
	"atax","bicg","gesummv","mvt", # Because these run only 1 threadblock
	"sad","bfs", # Because the memory accesses are atomic
	"syrk" # Curious results: require further investigation
	)

# Exclude some kernels from the results
kernel_excludes = c(
	"histo5","histo6","histo7","histo8","spmv2","lbm2","stencil2", # Because these are identical to earlier ones
	"mriq1","histo1","correlation3" # Because these have a very low execution time
	)

# Select which benchmarks to go where
firstplot = c("gemm","histo2","histo4","lbm1","stencil1","syrk","convolution2d","correlation1","correlation2","fdtd2d1","fdtd2d3")
secondplot = c("cutcp1","cutcp2","mriq2","mriq3","convolution3d","spmv1","bfs","fdtd2d2","histo3")


# Set the colours
purplish = "#550077" # [ 85,  0,119] lumi=26
blueish  = "#4765b1" # [ 71,101,177] lumi=100
lgray    = "#969696" # [150,150,150] lumi=150
redish   = "#d67568" # [214,117,104] lumi=136
greenish = "#9bd4ca" # [155,212,202] lumi=199
colourset = rep(c(blueish, redish, purplish, greenish), 3)

# Set the graph visuals
pchs = c(15, 5, 4, 6, 1, 2, 3, 5, 7, 8, 9, 10, 11)
linetypes = c(1, 2, 3, 4, 5, 6, 7, 8)
min = 0.0
max = 1.5

# Set the labels
xlabels = c("1/2", "2/3", "5/6", "1", "5/6", "2/3", "1/2")
ylabels = c("", "", "", "")
titles =  c("norm. performance", "norm. dyn. power (DFS)", "norm. total power (DFS)", "energy efficiency (DFS)",
            ""                 , "norm. dynamic power",    "norm. total power",       "energy efficiency")

##################################
## First part of the script (read the database)
##################################

# Read the data from file
fulldb = read.csv("results/database.csv")
print(fulldb)

##################################
## Second part of the script (create the graph)
##################################

# Iterate over the different plots
for (config in 1:4) {

	# Configure for microbenchmarks only (consise - paper version)
	if (config == 1) {
		pdf("results/microbenchmarks.pdf", height=2, width=8.6)
		par(mfrow=c(1, 4))
		types = c(99)
		pids = c(1,6,7,8)
		legendpids = c(1,6,7,8)
		db = fulldb[fulldb$type==0,]
	}
	# Configure for microbenchmarks only (full details)
	if (config == 2) {
		pdf("results/microbenchmarks_full.pdf", height=4.5, width=9)
		par(mfrow=c(2, 4))
		types = c(99)
		pids = c(1,2,3,4,5,6,7,8)
		legendpids = c(2,3,6,7)
		db = fulldb[fulldb$type==0,]
	}
	# Configure for real benchmarks (consise - paper version)
	if (config == 3) {
		pdf("results/benchmarks.pdf", height=2, width=8.6)
		par(mfrow=c(1, 4))
		types = c(1,2)
		pids = c(1,8)
		legendpids = c(8)
		db = fulldb[fulldb$type==1,]
	}
	# Configure for real benchmarks (full details)
	if (config == 4) {
		pdf("results/benchmarks_full.pdf", height=4.5, width=9)
		par(mfrow=c(2, 4))
		types = c(1,2)
		pids = c(1,2,3,4,5,6,7,8)
		legendpids = c(2,3,6,7)
		db = fulldb[fulldb$type==1,]
	}
	par(oma=c(0, 0, 0, 0))
	par(mar=c(4, 3, 2, 1))
	par(mgp=c(3, 0.7, 0))

	# Loop over the 2 sets of plots (low versus high computational intensities)
	for (s in types) {
		dbs = db
		if (s==1) { legendposy = 0.6; legendposx = 0.86; legendcols = 2; legendsize = 0.6 }
		if (s==2) { legendposy = 0.6; legendposx = 0.86; legendcols = 2; legendsize = 0.6 }
		if (s==99) { legendcols = 1; legendposx = 2.5; legendposy = 0.35; legendsize = 0.7 }
		
		# Loop over the plots
		for (pid in pids) {
			
			# Don't plot the performance graph again, but instead plot the computational intensities
			if (pid == 5) {
				par(new=F)
				plot(seq(1:7), c(0, 0, 0, 0, 0, 0, 0), axes=F, xlab="", ylab="", "n", ylim=c(min, max))
				text(2.85, 1.47, labels="comp. int. [ins/memory ins]", xpd=T)
				text(2.85, 0.57, labels="comp. int. [ins/offchip byte]", xpd=T)
				legend(0, 1.4, legdata5, lwd=1, ncol=2, col=legdata2, pch=legdata3, lty=legdata4, cex=0.7, xpd=T)
				legend(0, 0.5, legdata6, lwd=1, ncol=2, col=legdata2, pch=legdata3, lty=legdata4, cex=0.7, xpd=T)
				par(new=T)
			}
			else {
				
				# Create an initial graph with axis but with no data
				par(new=F)
				plot(seq(1:7), c(0, 0, 0, 0, 0, 0, 0), main=titles[pid], xlab="", ylab="", ylim=c(min, max), axes=F, "n")
				axis(2, at=c(0, 0.25, 0.5, 0.75, 1, 1.25, 1.5), las=2)
				axis(1, at=seq(1:7), labels=xlabels, las=1)
				
				# Add the gray lines/text/arrows
				abline(v=4, col=lgray, lty=2)
				text(4, min-0.60, labels="core scaling", col=lgray, cex=1, xpd=T, pos=2)
				text(4, min-0.60, labels="memory scaling", col=lgray, cex=1, xpd=T, pos=4)
				arrows(3.8, min-0.45, 1.9, min-0.45, col=lgray, length=0.05, xpd=T)
				arrows(4.2, min-0.45, 6.1, min-0.45, col=lgray, length=0.05, xpd=T)
				par(new=T)

				# Prepare the double-loop
				n = 1
				legdata1 = c(); legdata2 = c(); legdata3 = c(); legdata4 = c(); legdata5 = c(); legdata6 = c()

				# Split the data by name and kernel ID
				for (name in unique(db$name)) {
					if (!(name %in% bench_excludes)) {
						plotted_something = FALSE
						dbn = dbs[dbs$name==name,]
						k = 1
						for (kID in unique(dbn$kID)) {
							dbnk = dbn[dbn$kID==kID,]
							default = dbnk[dbnk$core==defaultcore & dbnk$mem==defaultmem,]
							
							# Get the name + number
							if ( length(unique(dbn$kID)) > 1 ) {
								namestring = p(name, kID)
							} else {
								namestring = name
							}
							
							# Test to which of the two plots this one should go
							included = TRUE
							if (s==1 && namestring %in% secondplot) {
								included = FALSE
							}
							if (s==2 && namestring %in% firstplot) {
								included = FALSE
							}
							
							#  Test if this one is not excluded
							if (included && !(namestring %in% kernel_excludes)) {
							
								# Select the data
								if (pid == 1 || pid == 5) { dataset = dbnk$performance
								} else if (pid == 2)      { dataset = dbnk$normdynpowerDFS
								} else if (pid == 3)      { dataset = dbnk$normtotalpowerDFS
								} else if (pid == 4)      { dataset = dbnk$effDFS
								} else if (pid == 6)      { dataset = dbnk$normdynpowerDVFS
								} else if (pid == 7)      { dataset = dbnk$normtotalpowerDVFS
								} else if (pid == 8)      { dataset = dbnk$effDVFS }
								
								# Plot the data
								plot(seq(1:7), dataset, col=colourset[n], pch=pchs[n], lty=linetypes[k], lwd=1, ylim=c(min, max), axes=F, xlab="", ylab="", "b", xpd=T)
								par(new=T)
								
								# Store data for the legend
								if (pid %in% c(2,6)) {
									powerstring = p(p(" (", as.integer(default$dynpower)), "W)")
								} else if (pid %in% c(3,7)) {
									powerstring = p(p(" (", as.integer(default$totalpower)), "W)")
								} else if (config %in% c(1)) {
									powerstring = ""
								} else {
									powerstring = p(p(" (", default$int_bytes), ")")
								}
								intensity1string = p(p(" = ", default$int_ins), "")
								intensity2string = p(p(" = ", default$int_bytes), "")
								legdata1 = c(legdata1, p(namestring, powerstring))
								legdata2 = c(legdata2, colourset[n])
								legdata3 = c(legdata3, pchs[n])
								legdata4 = c(legdata4, linetypes[k])
								legdata5 = c(legdata5, p(namestring, intensity1string))
								legdata6 = c(legdata6, p(namestring, intensity2string))
								plotted_something = TRUE
							}
							# Next item
							k = k + 1
						}
						if (plotted_something) {
							n = n + 1
						}
					}
				}
				
				# Add a legend
				if (pid %in% legendpids) {
					legend(legendposx, legendposy, legdata1, lwd=1, ncol=legendcols, col=legdata2, pch=legdata3, lty=legdata4, cex=legendsize,
					       box.lwd=0, box.col ="white", bg="white", xpd=T)
				}

				# Add an extra line
				if (config == 3 && s==1 && pid==8) {
					abline(v=7.6, col=lgray, lty=1, lwd=3, xpd=T)
				}
				
				# Add circles for the microbenchmarks
				if (config == 1 && pid == 8) {
					text(1.05, 1.56, labels="58%", col=lgray, cex=1, xpd=T, pos=4)
					text(7, 1.27, labels="2%", col=lgray, cex=1, xpd=T, pos=1)
				}
			}
		}
	}
}

##################################