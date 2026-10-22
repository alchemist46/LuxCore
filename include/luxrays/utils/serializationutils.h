/***************************************************************************
 * Copyright 1998-2020 by authors (see AUTHORS.txt)                        *
 *                                                                         *
 *   This file is part of LuxCoreRender.                                   *
 *                                                                         *
 * Licensed under the Apache License, Version 2.0 (the "License");         *
 * you may not use this file except in compliance with the License.        *
 * You may obtain a copy of the License at                                 *
 *                                                                         *
 *     http://www.apache.org/licenses/LICENSE-2.0                          *
 *                                                                         *
 * Unless required by applicable law or agreed to in writing, software     *
 * distributed under the License is distributed on an "AS IS" BASIS,       *
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.*
 * See the License for the specific language governing permissions and     *
 * limitations under the License.                                          *
 ***************************************************************************/

#ifndef _SLG_SERIALIZATIONUTILS_H
#define	_SLG_SERIALIZATIONUTILS_H

#include <string>
#include <deque>
#include <set>
#include <vector>

#include <filesystem>
#include <fstream>
#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/filter/gzip.hpp>

#include <boost/archive/basic_archive.hpp>

#include <boost/serialization/serialization.hpp>
#include <boost/serialization/version.hpp>
#include <boost/serialization/access.hpp>
#include <boost/serialization/split_free.hpp>
#include <boost/serialization/array.hpp>
#include <boost/serialization/level.hpp>
#include <boost/serialization/export.hpp>
#include <boost/serialization/split_member.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/assume_abstract.hpp>
#include <boost/serialization/tracking.hpp>
#include <boost/serialization/unique_ptr.hpp>

#include <boost/serialization/deque.hpp>
#include <boost/serialization/set.hpp>
#include <boost/serialization/vector.hpp>

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

#include "luxrays/luxrays.h"

namespace luxrays {

typedef boost::archive::binary_oarchive LuxOutputArchive;
typedef boost::archive::binary_iarchive LuxInputArchive;
typedef boost::archive::text_oarchive LuxOutputArchiveText;
typedef boost::archive::text_iarchive LuxInputArchiveText;

// Saves an object that something else owns as the std::unique_ptr<T> its
// loader reads (a unique_ptr has its own class information in the archive,
// so neither the object nor a raw pointer to it can be read back that way),
// without the temporary unique_ptr deleting it
template <class Archive, class T> void SaveAsUniquePtr(Archive &ar, const T &obj) {
	std::unique_ptr<T> ptr(const_cast<T *>(&obj));
	try {
		ar << ptr;
	} catch (...) {
		ptr.release();
		throw;
	}
	ptr.release();
}

class SerializationOutputFile {
public:
	SerializationOutputFile(const std::string &fileName);
	virtual ~SerializationOutputFile();

	LuxOutputArchive &GetArchive();

	bool IsGood();
	std::streampos GetPosition();
	void Flush();

private:
	std::ofstream outFile;
	boost::iostreams::filtering_ostream outStream;
	LuxOutputArchive *outArchive;
};

class SerializationInputFile {
public:
	SerializationInputFile(const std::string &fileName);
	virtual ~SerializationInputFile();

	LuxInputArchive &GetArchive();

	bool IsGood();

private:
	std::ifstream inFile;
	boost::iostreams::filtering_istream inStream;
	LuxInputArchive *inArchive;
};

}  // namespace luxrays


namespace boost {
namespace serialization {

// Helper for std::reference_wrapper
template<class Archive, typename T>
void serialize(Archive & ar, std::reference_wrapper<T> & ref, const unsigned int version) {
    // Serialize the referenced object
    ar & ref.get();
}

} // namespace serialization
} // namespace boost


#endif	/* _SLG_SERIALIZATIONUTILS_H */
// vim: autoindent noexpandtab tabstop=4 shiftwidth=4
