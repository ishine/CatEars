subdirs: $(SUBDIRS)
	for x in $(SUBDIRS); do $(MAKE) -C $$x; done

CC_SOURCES = $(filter %.cc, $(SOURCES))
CC_OBJECTS = $(patsubst %.cc, %.o, $(CC_SOURCES))

OBJECTS = $(CC_OBJECTS)

library: $(LIBRARY)

executable: $(EXECUTABLE)

$(LIBRARY): $(OBJECTS)
	$(AR) -cru $(LIBRARY) $(OBJECTS)
	$(RANLIB) $(LIBRARY)

$(EXECUTABLE): $(OBJECTS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $(OBJECTS) $(LIBS) -o $(EXECUTABLE)

.cc.o:
	$(CXX) $(CXXFLAGS) -c $*.cc -o $*.o

clean:
	-rm *.o depend.mk
	-for x in $(SUBDIRS); do $(MAKE) -C $$x clean; done
	
depend:
	-$(CXX) -M $(CXXFLAGS) *.cc > depend.mk
	-for x in $(SUBDIRS); do $(MAKE) -C $$x depend; done

#depend.mk: depend
-include depend.mk
