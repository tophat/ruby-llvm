module LLVM
  class Value
    # @private
    def self.from_ptr(ptr)
      return if ptr.null?
      val = allocate
      val.instance_variable_set(:@ptr, ptr)
      val
    end

    # @private
    def to_ptr
      @ptr
    end

    # Checks if this value is equal to another.
    # @param [LLVM::Value] other The value to compare this one with
    def ==(other)
      case other
      when LLVM::Value
        @ptr == other.to_ptr
      else
        false
      end
    end
    
    def hash
      @ptr.address.hash
    end

    # Checks if this value is equal to another.
    # @param [LLVM::Value] other The value to compare this one with
    def eql?(other)
      other.instance_of?(self.class) && self == other
    end

    # Gets this value's type. This is abstract and is overidden by its subclasses.
    def self.type
      raise NotImplementedError, "#{self.name}.type() is abstract."
    end

    # @private
    def self.to_ptr
      type.to_ptr
    end

    # Gets the value's type.
    # @return [LLVM::Type] The type of this value
    def type
      Type.from_ptr(C.type_of(self), nil)
    end

    # Gets the value's name.
    # @return [String] The name of this value in LLVM IR
    def name
      C.get_value_name(self)
    end

    # Sets the value's name.
    # @param [String] str The new name
    def name=(str)
      C.set_value_name(self, str)
      str
    end

    # Print the value's IR to stdout.
    def dump
      C.dump_value(self)
    end

    # Gets whether the value is constant.
    # @return [Boolean] The resulting true/false value
    def constant?
      case C.is_constant(self)
      when 0 then false
      when 1 then true
      end
    end

    # Gets whether the value is null.
    # @return [Boolean] The resulting true/false value
    def null?
      case C.is_null(self)
      when 0 then false
      when 1 then true
      end
    end

    # Gets whether the value is undefined.
    # @return [Boolean] The resulting true/false value
    def undefined?
      case C.is_undef(self)
      when 0 then false
      when 1 then true
      end
    end

    # Adds an attribute to this value's attributes.
    # @param [Symbol] attr The attribute to add
    def add_attribute(attr)
      C.add_attribute(self, attr)
    end
  end

  class Argument < Value
  end

  class BasicBlock < Value
    # Creates a basic block in the given function with an optional name.
    # @param [LLVM::Function] fun The function containing the new basic block
    # @param [String] name An optional name for the new basic block
    # @return [LLVM::BasicBlock] The new basic block
    def self.create(fun = nil, name = "")
      self.from_ptr(C.append_basic_block(fun, name))
    end

    # Build this basic block with the given builder which is yielded.
    # @param [LLVM::Builder] builder The builder to build with, creates a new one if nil
    def build(builder = nil)
      if builder.nil?
        builder = Builder.new
        builder.position_at_end(self)
        yield builder
        builder.dispose
      else
        builder.position_at_end(self)
        yield builder
      end
    end
    
    # Deletes this basic block.
    def dispose
      C.delete_basic_block(self)
    end

    # Gets the parent of this basic block.
    # @return [LLVM::Function] The parent function
    def parent
      fp = C.get_basic_block_parent(self)
      LLVM::Function.from_ptr(fp) unless fp.null?
    end

    # Gets the next basic block in the sequence.
    # @return [LLVM::BasicBlock] The next basic block
    def next
      ptr = C.get_next_basic_block(self)
      BasicBlock.from_ptr(ptr) unless ptr.null?
    end

    # Gets the previous basic block in the sequence.
    # @return [LLVM::BasicBlock] The previous basic block
    def previous
      ptr = C.get_previous_basic_block(self)
      BasicBlock.from_ptr(ptr) unless ptr.null?
    end
    
    # Moves this basic block before block.
    # @param [LLVM::BasicBlock] block The basic block that this block will be placed before.
    def move_before(block)
      C.move_basic_block_before(self, block)
    end

    # Moves this basic block after block.
    # @param [LLVM::BasicBlock] block The basic block that this block will be placed after 
    def move_after(block)
      C.move_basic_block_after(self, block)
    end

    def first_instruction  # deprecated
      instructions.first
    end

    def last_instruction  # deprecated
      instructions.last
    end
    
    # Checks if this basic block contains any instructions. 
    # @return [Boolean] The resulting true/false value
    def empty?
      return instructions.first.nil?
    end

    # Gets an Enumerable of the instructions in the current block.
    # @return [LLVM::BasicBlock::InstructionCollection] The enumerable
    def instructions
      @instructions ||= InstructionCollection.new(self)
    end

    # An Enumerable containing the instructions in a LLVM::BasicBlock.
    class InstructionCollection
      include Enumerable

      # @private
      def initialize(block)
        @block = block
      end

      # Iterates through each instruction in the collection.
      def each
        return to_enum :each unless block_given?
        inst, last = first, last

        while inst
          yield inst
          break if inst == last
          inst = inst.next
        end

        self
      end

      # Gets the first instruction in the collection.
      # @return [LLVM::Instruction] The first instruction
      def first
        ptr = C.get_first_instruction(@block)
        LLVM::Instruction.from_ptr(ptr) unless ptr.null?
      end

      # Gets the last instruction in the collection.
      # @return [LLVM::Instruction] The last instruction
      def last
        ptr = C.get_last_instruction(@block)
        LLVM::Instruction.from_ptr(ptr) unless ptr.null?
      end
    end
  end

  class User < Value
    # Gets an Enumerable of the operands in this user.
    # @return [LLVM::User::OperandCollection] The enumerable.
    def operands
      @operand_collection ||= OperandCollection.new(self)
    end

    # An Enumerable containing the operands in a LLVM::User.
    class OperandCollection
      include Enumerable

      # @private
      def initialize(user)
        @user = user
      end

      # Gets the operand at the given index.
      # @return [LLVM::Value] The operand.
      def [](i)
        ptr = C.get_operand(@user, i)
        Value.from_ptr(ptr) unless ptr.null?
      end

      # Sets the operand at the given index.
      # @param [Integer] i The index at which to place the operand.
      # @param [LLVM::Value] v The operand to be placed.
      def []=(i, v)
        C.set_operand(@user, i, v)
      end

      # Gets the number of operands in the collection.
      # @return [Integer] The number of operands.
      def size
        C.get_num_operands(@user)
      end

      # Iterates through each operand in the collection.
      def each
        return to_enum :each unless block_given?
        0.upto(size-1) { |i| yield self[i] }
        self
      end
    end
  end

  class Constant < User
    # Creates a null constant of type.
    # @param [LLVM::Type] type The type of the constant
    def self.null(type)
      from_ptr(C.const_null(type))
    end

    # Creates a undefined constant of type.
    # @param [LLVM::Type] type The type of the constant
    def self.undef(type)
      from_ptr(C.get_undef(type))
    end

    # Creates a null pointer constant of type.
    # @param [LLVM::Type] type The type of the constant
    def self.null_ptr(type)
      from_ptr(C.const_pointer_null(type))
    end

    # Bitcast this constant to type.
    # @param [LLVM::Type] type The type to bitcast to
    def bitcast_to(type)
      ConstantExpr.from_ptr(C.const_bit_cast(self, type))
    end

    # Gets the element pointer at the given indices of the constant.
    # @param [Array<LLVM::Value>] indices Ruby array of LLVM::Value representing
    #   indices into the aggregate
    # @return [LLVM::Instruction] The resulting pointer
    # @see http://llvm.org/docs/GetElementPtr.html
    def gep(*indices)
      indices = Array(indices)
      FFI::MemoryPointer.new(FFI.type_size(:pointer) * indices.size) do |indices_ptr|
        indices_ptr.write_array_of_pointer(indices)
        return ConstantExpr.from_ptr(
          C.const_gep(self, indices_ptr, indices.size))
      end
    end
  end

  # @private
  module Support
    def allocate_pointers(size_or_values, &block)
      if size_or_values.is_a?(Integer)
        raise ArgumentError, 'block not given' unless block_given?
        size = size_or_values
        values = (0...size).map { |i| yield i }
      else
        values = size_or_values
        size = values.size
      end
      FFI::MemoryPointer.new(:pointer, size).write_array_of_pointer(values)
    end

    module_function :allocate_pointers
  end

  class ConstantArray < Constant
    # Creates a new constant array containing a string.
    # @param [String] str Ruby string for the array to contain
    # @param [Boolean] null_terminate
    # @return [LLVM::ConstantArray] The resulting array
    def self.string(str, null_terminate = true)
      from_ptr(C.const_string(str, str.length, null_terminate ? 0 : 1))
    end

    # Creates a new constant array.
    # Usage: ConstantArray.const(type, 3) {|i| ... } or ConstantArray.const(type, [...])
    # @param [LLVM::Type] type The type of the new array
    # @param [Integer or Array] size_or_values Either the size of the array or its values
    # @param [Proc] block A block returning the value for each element if a size is passed
    # @return [LLVM::ConstantArray] The resulting array
    def self.const(type, size_or_values, &block)
      vals = LLVM::Support.allocate_pointers(size_or_values, &block)
      from_ptr C.const_array(type, vals, vals.size / vals.type_size)
    end
  end

  class ConstantExpr < Constant
  end

  class ConstantInt < Constant
    # Creates a ConstantInt in which all bits arre set to 1.
    # @return [LLVM::ConstantInt] The resulting int
    def self.all_ones
      from_ptr(C.const_all_ones(type))
    end

    # Creates a new constant int from a Ruby integer.
    # @param [Integer] n Ruby integer for the constant to contain
    # @param [Boolean] signed Whether or not this integer can be negative
    def self.from_i(n, signed = true)
      from_ptr(C.const_int(type, n, signed ? 1 : 0))
    end

    # Creates a new constant int from a Ruby string
    # @param [String] str Ruby string containing the integer
    # @param [Integer] radix The integer radix
    def self.parse(str, radix = 10)
      from_ptr(C.const_int_of_string(type, str, radix))
    end

    # Sign Negation.
    # @return [LLVM::ConstantInt] The resulting integer of the opposite sign
    def -@
      self.class.from_ptr(C.const_neg(self))
    end

    # Bitwise negation.
    # @return [LLVM::ConstantInt] The resulting integer
    def not
      self.class.from_ptr(C.const_not(self))
    end

    # Addition.
    # @param [LLVM::ConstantInt] rhs The integer to add to this one
    # @return [LLVM::ConstantInt] The resulting sum
    def +(rhs)
      self.class.from_ptr(C.const_add(self, rhs))
    end

    # "No signed wrap" addition.
    # @param [LLVM::ConstantInt] rhs The integer to add to this one
    # @return [LLVM::ConstantInt] The resulting sum
    # @see http://llvm.org/docs/LangRef.html#i_add
    def nsw_add(rhs)
      self.class.from_ptr(C.const_nsw_add(self, rhs))
    end

    # Multiplication.
    # @param [LLVM::ConstantInt] rhs The integer to multiply with this one
    # @return [LLVM::ConstantInt] The resulting product
    def *(rhs)
      self.class.from_ptr(C.const_mul(self, rhs))
    end

    # Unsigned division.
    # @param [LLVM::ConstantInt] rhs The positive integer to divide this one by
    # @return [LLVM::ConstantInt] The resulting quotient
    def udiv(rhs)
      self.class.from_ptr(C.const_u_div(self, rhs))
    end

    # Signed division.
    # @param [LLVM::ConstantInt] rhs The integer to divide this one by
    # @return [LLVM::ConstantInt] The resulting quotient
    def /(rhs)
      self.class.from_ptr(C.const_s_div(self, rhs))
    end

    # Unsigned remainder, equivalent to modulo (%).
    # @param [LLVM::ConstantInt] rhs The integer to divide this one by
    # @return [LLVM::ConstantInt] The resulting positive remainder
    def urem(rhs)
      self.class.from_ptr(C.const_u_rem(self, rhs))
    end

    # Signed remainder, equivalent to modulo (%).
    # @param [LLVM::ConstantInt] rhs The integer to divide this one by
    # @return [LLVM::ConstantInt] The resulting remainder
    def rem(rhs)
      self.class.from_ptr(C.const_s_rem(self, rhs))
    end

    # Bitwise AND (&).
    # @param [LLVM::ConstantInt] rhs The integer to AND with
    # @return [LLVM::ConstantInt] The resulting ConstantInt
    def and(rhs)
      self.class.from_ptr(C.const_and(self, rhs))
    end

    # Bitwise OR (|).
    # @param [LLVM::ConstantInt] rhs The integer to OR with
    # @return [LLVM::ConstantInt] The resulting integer
    def or(rhs)
      self.class.from_ptr(C.const_or(self, rhs))
    end

    # Bitwise XOR (^).
    # @param [LLVM::ConstantInt] rhs The integer to XOR with
    # @return [LLVM::ConstantInt] The resulting integer
    def xor(rhs)
      self.class.from_ptr(C.const_xor(self, rhs))
    end

    # Integer comparison using the predicate specified via the first parameter.
    # Predicate can be any of:
    #   :eq  - equal to
    #   :ne  - not equal to
    #   :ugt - unsigned greater than
    #   :uge - unsigned greater than or equal to
    #   :ult - unsigned less than
    #   :ule - unsigned less than or equal to
    #   :sgt - signed greater than
    #   :sge - signed greater than or equal to
    #   :slt - signed less than
    #   :sle - signed less than or equal to
    # @param [Symbol] pred One of the above predicates
    # @param [LLVM::ConstantInt] rhs The integer to compare to
    # @return [LLVM::ConstantInt] The resulting integer
    def icmp(pred, rhs)
      self.class.from_ptr(C.const_i_cmp(pred, self, rhs))
    end

    # Shift left.
    # @param [LLVM::ConstantInt] bits The number of bits to shift left
    # @return [LLVM::ConstantInt] The resulting integer
    def <<(bits)
      self.class.from_ptr(C.const_shl(self, bits))
    end

    # Shift right.
    # @param [LLVM::ConstantInt] bits The number of bits to shift right
    # @return [LLVM::ConstantInt] The resulting integer
    def >>(bits)
      self.class.from_ptr(C.const_l_shr(self, bits))
    end

    # Arithmatic shift right.
    # @param [LLVM::ConstantInt] bits The number of bits to shift right
    # @return [LLVM::ConstantInt] The resulting integer
    def ashr(bits)
      self.class.from_ptr(C.const_a_shr(self, bits))
    end
  end

  # @private
  def LLVM.const_missing(const)
    case const.to_s
    when /Int(\d+)/
      width = $1.to_i
      name  = "Int#{width}"
      eval <<-KLASS
        class #{name} < ConstantInt
          def self.type
            Type.from_ptr(C.int_type(#{width}), :integer)
          end
        end
      KLASS
      const_get(name)
    else
      super
    end
  end

  # Native integer type
  bits = FFI.type_size(:int) * 8
  ::LLVM::Int = const_get("Int#{bits}")

  # Creates a LLVM::Int (subclass of ConstantInt) at the NATIVE_INT_SIZE from a integer.
  # @param [Integer] val Ruby integer for the LLVM::Int to contain
  # @return [LLVM::Int] The resulting integer
  def LLVM.Int(val)
    case val
    when LLVM::ConstantInt then val
    when Integer then Int.from_i(val)
    end
  end
  
  # Boolean values
  ::LLVM::TRUE = ::LLVM::Int1.from_i(-1)
  ::LLVM::FALSE = ::LLVM::Int1.from_i(0)

  class ConstantReal < Constant
    # Creates a ConstantReal from a Ruby float.
    # @param [Float] n Ruby float for the real to contain
    def self.from_f(n)
      from_ptr(C.const_real(type, n))
    end

    # Creates a ConstantReal from a Ruby string
    # @param [LLVM::Type] type The type of real to create (Float, Double, etc.)
    # @param [String] str Ruby string containing the real
    def self.parse(type, str)
      from_ptr(C.const_real_of_string(type, str))
    end

    # Sign Negation.
    # @return [LLVM::ConstantReal] The resulting real of the opposite sign
    def -@
      self.class.from_ptr(C.const_f_neg(self))
    end

    # Addition.
    # @param [LLVM::ConstantReal] rhs The real to add to this one
    # @return [LLVM::ConstantReal] The resulting sum
    def +(rhs)
      self.class.from_ptr(C.const_f_add(self, rhs))
    end

    # Multiplication.
    # @param [LLVM::ConstantReal] rhs The real to multiply with this one
    # @return [LLVM::ConstantReal] The resulting product
    def *(rhs)
      self.class.from_ptr(C.const_f_mul(self, rhs))
    end

    # Divison.
    # @param [LLVM::ConstantReal] rhs The real to divide this one by
    # @return [LLVM::ConstantReal] The resulting quotient
    def /(rhs)
      self.class.from_ptr(C.const_f_div(self, rhs))
    end

    # Remainder, equivalent to modulo (%).
    # @param [LLVM::ConstantReal] rhs The real to divide this one by
    # @return [LLVM::ConstantReal] The resulting remainder
    def rem(rhs)
      self.class.from_ptr(C.const_f_rem(self, rhs))
    end

    # Floating point comparison using the predicate specified via the first
    # parameter. Predicate can be any of:
    #   :ord  - ordered
    #   :uno  - unordered: isnan(X) | isnan(Y)
    #   :oeq  - ordered and equal to
    #   :oeq  - unordered and equal to
    #   :one  - ordered and not equal to
    #   :one  - unordered and not equal to
    #   :ogt  - ordered and greater than
    #   :uge  - unordered and greater than or equal to
    #   :olt  - ordered and less than
    #   :ule  - unordered and less than or equal to
    #   :oge  - ordered and greater than or equal to
    #   :sge  - unordered and greater than or equal to
    #   :ole  - ordered and less than or equal to
    #   :sle  - unordered and less than or equal to
    #   :true - always true
    #   :false- always false
    # @param [Symbol] pred One of the above predicates
    # @param [LLVM::ConstantReal] rhs The real to compare to
    # @return [LLVM::ConstantReal] The resulting ral
    def fcmp(pred, rhs)
      self.class.from_ptr(C.llmv_const_f_cmp(pred, self, rhs))
    end
  end

  class Float < ConstantReal
    # Gets the float type.
    # @return [LLVM::Type] The float type. 
    def self.type
      Type.from_ptr(C.float_type, :float)
    end
  end

  # Create a LLVM::Float from a Ruby float.
  # @param [Float] val Ruby float for the LLVM::Float to contain
  # @return [LLVM::Float] The resulting float
  def LLVM.Float(val)
    Float.from_f(val)
  end

  class Double < ConstantReal
    # Gets the double type.
    # @return [LLVM::Type] The double type.
    def self.type
      Type.from_ptr(C.double_type, :double)
    end
  end

  # Create a LLVM::Double from a Ruby float.
  # @param [Float] val Ruby float for the LLVM::Double to contain
  # @return [LLVM::Double] The resulting double
  def LLVM.Double(val)
    Double.from_f(val)
  end

  class ConstantStruct < Constant
    # Creates a new constant struct.
    # Usage: ConstantStruct.const(size) {|i| ... } or ConstantStruct.const([...])
    # @param [Integer or Array] size_or_values Either the number of elements in the struct 
    #   or the elements themselves.
    # @param [Proc] block A block returning the value for each element if a size is passed
    # @return [LLVM::ConstantStruct] The resulting struct
    def self.const(size_or_values, packed = false, &block)
      vals = LLVM::Support.allocate_pointers(size_or_values, &block)
      from_ptr C.const_struct(vals, vals.size / vals.type_size, packed ? 1 : 0)
    end
    
    # Same as ::const, but creates the struct with a pre-existing type.
    # @param [LLVM::Type] type The type of the new struct
    # @param [Integer or Array] size_or_values Either the number of elements in the struct 
    #   or the elements themselves.
    # @param [Proc] block A block returning the value for each element if a size is passed
    # @return [LLVM::ConstantStruct] The resulting struct
    def self.typed_const(type, size_or_values, &block)
      vals = LLVM::Support.allocate_pointers(size_or_values, &block)
      from_ptr(C.const_named_struct(type, vals, vals.size / vals.type_size))
    end
  end

  class ConstantVector < Constant
    # Creates a ConstantVector in which all bits are set to 1.
    # @return [LLVM::ConstantVector] The resulting vector
    def self.all_ones
      from_ptr(C.const_all_ones(type))
    end

    # Creates a new constant vector.
    # Usage: ConstantVector.const(size) {|i| ... } or ConstantVector.const([...])
    # @param [Integer or Array] size_or_values Either the number of elements in the vector 
    #   or the elements themselves.
    # @param [Proc] block A block returning the value for each element if a size is passed
    # @return [LLVM::ConstantVector] The resulting vector
    def self.const(size_or_values, &block)
      vals = LLVM::Support.allocate_pointers(size_or_values, &block)
      from_ptr(C.const_vector(vals, vals.size / vals.type_size))
    end
  end

  class GlobalValue < Constant
    # Checks whether this global value is a declaration.
    # @return [Integer] A 0 or 1 int. 1 for true and 0 for false
    def declaration?
      C.is_declaration(self)
    end

    # Gets the global value's linkage.
    # @return [Symbol] The value's linkage
    def linkage
      C.get_linkage(self)
    end

    # Sets the global value's linkage.
    # @param [Symbol] linkage The value's new linkage
    def linkage=(linkage)
      C.set_linkage(self, linkage)
    end

    # Gets the global value's section.
    # @return [String] The value's section
    def section
      C.get_section(self)
    end

    # Sets the global value's section.
    # @param [String] section The value's new section
    def section=(section)
      C.set_section(self, section)
    end

    # Gets the global value's visibility (one of :default, :hidden, or :protected). 
    # @return [Symbol] The value's visibility
    def visibility
      C.get_visibility(self)
    end

    # Sets the global value's visibility (one of :default, :hidden, or :protected).
    # @param [Symbol] viz The value's new visibility 
    def visibility=(viz)
      C.set_visibility(self, viz)
    end
    
    # Gets the global value's alignment
    # @return [Integer] The value's alignement (positive)
    def alignment
      C.get_alignment(self)
    end

    # Gets the global value's alignment
    # @param [Integer] bytes The value's new alignement (positive)
    def alignment=(bytes)
      C.set_alignment(self, bytes)
    end

    # Gets the global value's initializer.
    # @return [LLVM::Value] The value's initializer
    def initializer
      Value.from_ptr(C.get_initializer(self))
    end

    # Sets the global value's initializer.
    # @param [LLVM::Value] val The value's new initializer
    def initializer=(val)
      C.set_initializer(self, val)
    end

    # Checks whether this global value is a global constant.
    # @return [Integer] A 0 or 1 int. 1 for true and 0 for false
    def global_constant?
      C.is_global_constant(self)
    end

    # Sets whether this global value is a global constant.
    # @param [Integer] flag A 0 or 1 int. 1 for true and 0 for false
    def global_constant=(flag)
      C.set_global_constant(self, flag)
    end
  end

  class Function < GlobalValue
    # Sets the function's calling convention and returns it.
    # @param [Integer] conv The function's new calling convention
    # @return [Integer] The new calling convention
    def call_conv=(conv)
      C.set_function_call_conv(self, conv)
      conv
    end

    # Adds the given attribute to the function.
    # @param [Symbol] attr The attribute to be added
    def add_attribute(attr)
      C.add_function_attr(self, attr)
    end

    # Removes the given attribute from the function.
    # @param [Symbol] attr The attribute to be removed
    def remove_attribute(attr)
      C.remove_function_attr(self, attr)
    end

    # Returns an Enumerable of the BasicBlocks in this function.
    # @return [LLVM::Function::BasicBlockCollection] The enumerable
    def basic_blocks
      @basic_block_collection ||= BasicBlockCollection.new(self)
    end

    # Gets the function's type.
    # @return [LLVM::FunctionType] The function's type.
    def type
      Type.from_ptr(C.type_of(self), :function)
    end

    # An Enumerable containing the basic blocks in a LLVM::Function.
    class BasicBlockCollection
      include Enumerable

      # @private
      def initialize(fun)
        @fun = fun
      end

      # Gets the number of basic blocks in the collection.
      # @return [Integer] The number of basic blocks
      def size
        C.count_basic_blocks(@fun)
      end

      # Iterates through each basic block in the collection.
      def each
        return to_enum :each unless block_given?

        ptr = C.get_first_basic_block(@fun)
        0.upto(size-1) do |i|
          yield BasicBlock.from_ptr(ptr)
          ptr = C.get_next_basic_block(ptr)
        end

        self
      end

      # Adds a basic block with an optional name to the end of the collection and returns it.
      # @param [String] name The optional name of the block in LLVM IR
      # @return [LLVM::BasicBlock] The new block
      def append(name = "")
        BasicBlock.create(@fun, name)
      end
      
      # Inserts a basic block before the given block and returns it.
      # @param [String] block The block which the new block should be placed before
      # @param [String] name The optional name of the block in LLVM IR
      # @return [LLVM::BasicBlock] The new block
      def insert(block, name="") 
        BasicBlock.from_ptr(C.insert_basic_block(block_after, name))
      end

      # Gets the entry basic block in the collection. This is the block the function starts on.
      # @return [LLVM::BasicBlock] The entry block
      def entry
        BasicBlock.from_ptr(C.get_entry_basic_block(@fun))
      end

      # Gets the first basic block in the collection.
      # @return [LLVM::BasicBlock] The first basic block
      def first
        ptr = C.get_first_basic_block(@fun)
        BasicBlock.from_ptr(ptr) unless ptr.null?
      end

      # Gets the last basic block in the collection.
      # @return [LLVM::BasicBlock] The last basic block
      def last
        ptr = C.get_last_basic_block(@fun)
        BasicBlock.from_ptr(ptr) unless ptr.null?
      end
    end

    # Returns an Enumerable of the parameters in the function.
    # @return [LLVM::Function::ParameterCollection] The enumerable
    def params
      @parameter_collection ||= ParameterCollection.new(self)
    end

    # An Enumerable containing the parameters of a LLVM::Function.
    class ParameterCollection
      # @private
      def initialize(fun)
        @fun = fun
      end

      # Gets the parameter at the given index.
      # @return [LLVM::Value] The paramter at the given index
      def [](i)
        sz = self.size
        i = sz + i if i < 0
        return unless 0 <= i && i < sz
        Value.from_ptr(C.get_param(@fun, i))
      end

      # Returns the number of paramters in the collection.
      # @return [Integer] The size
      def size
        C.count_params(@fun)
      end

      include Enumerable

      # Iteraters through each parameter in the collection.
      def each
        return to_enum :each unless block_given?
        0.upto(size-1) { |i| yield self[i] }
        self
      end
    end
  end

  class GlobalAlias < GlobalValue
  end

  class GlobalVariable < GlobalValue
    # Gets the global variable's initializer.
    # @return [LLVM::Value] The variable's initializer
    def initializer
      Value.from_ptr(C.get_initializer(self))
    end

    # Sets the global variable's initializer.
    # @param [LLVM::Value] The variable's new initializer
    def initializer=(val)
      C.set_initializer(self, val)
    end

    # Gets whether the value is thread local.
    # @return [Boolean] The resulting true/false value
    def thread_local?
      case C.is_thread_local(self)
      when 0 then false
      else true
      end
    end

    # Sets whether the value is thread local.
    # @param [Boolean] A true/false value
    def thread_local=(local)
      C.set_thread_local(self, local ? 1 : 0)
    end
  end

  class Instruction < User
    # Gets the parent of the instruction.
    # @return [LLVM::BasicBlock] The parent.
    def parent
      ptr = C.get_instruction_parent(self)
      LLVM::BasicBlock.from_ptr(ptr) unless ptr.null?
    end

    # Gets the next instruction after this one.
    # @return [LLVM::Instruction] The next instruction.
    def next
      ptr = C.get_next_instruction(self)
      LLVM::Instruction.from_ptr(ptr) unless ptr.null?
    end

    # Gets the previous instruction before this one.
    # @return [LLVM::Instruction] The previous instruction.
    def previous
      ptr = C.get_previous_instruction(self)
      LLVM::Instruction.from_ptr(ptr) unless ptr.null?
    end
  end

  class CallInst < Instruction
    # Sets the calling convention and returns it.
    # @param [Integer] conv The new calling convention
    # @return [Integer] The new calling convention
    def call_conv=(conv)
      C.set_instruction_call_conv(self, conv)
      conv
    end

    # Gets the call instance's calling convention.
    # @return [Integer] The calling convention
    def call_conv
      C.get_instruction_call_conv(self)
    end
  end

  class Phi < Instruction
    # Adds incoming branches to a phi node.
    # @param [Hash{LLVM::BasicBlock => LLVM::Value}] incoming A hash mapping of
    #   basic blocks to a corresponding value. If the phi node is jumped to
    #   from a given basic block, the phi instruction takes on its
    #   corresponding value.
    def add_incoming(incoming)
      blks = incoming.keys
      vals = incoming.values_at(*blks)
      size = incoming.size

      FFI::MemoryPointer.new(FFI.type_size(:pointer) * size) do |vals_ptr|
        vals_ptr.write_array_of_pointer(vals)
        FFI::MemoryPointer.new(FFI.type_size(:pointer) * size) do |blks_ptr|
          blks_ptr.write_array_of_pointer(blks)
          C.add_incoming(self, vals_ptr, blks_ptr, vals.size)
        end
      end

      nil
    end
  end

  class SwitchInst < Instruction
    # Adds a case to a switch instruction. 
    # @param [LLVM::Value] val The value to match on
    # @param [LLVM::BasicBlock] block The basic block to execute if matched
    def add_case(val, block)
      C.add_case(self, val, block)
    end
  end
end
